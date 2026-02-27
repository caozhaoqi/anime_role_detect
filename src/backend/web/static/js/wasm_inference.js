/**
 * WebAssembly推理模块
 * 使用ONNX Runtime Web进行浏览器端推理
 */

class WasmInference {
    constructor() {
        this.session = null;
        this.isInitialized = false;
        this.inputName = null;
        this.outputName = null;
    }

    async initialize(modelPath) {
        try {
            const ort = window.ort;
            if (!ort) {
                throw new Error('ONNX Runtime Web未加载');
            }

            this.session = await ort.InferenceSession.create(modelPath);
            
            const inputNames = this.session.inputNames;
            const outputNames = this.session.outputNames;
            
            if (inputNames.length > 0) {
                this.inputName = inputNames[0];
            }
            if (outputNames.length > 0) {
                this.outputName = outputNames[0];
            }
            
            this.isInitialized = true;
            console.log('WebAssembly模型初始化成功');
            console.log('输入名称:', this.inputName);
            console.log('输出名称:', this.outputName);
            
            return true;
        } catch (error) {
            console.error('WebAssembly模型初始化失败:', error);
            throw error;
        }
    }

    async classifyImage(imageElement) {
        if (!this.isInitialized) {
            throw new Error('模型未初始化');
        }

        try {
            const ort = window.ort;
            
            const imageData = this.preprocessImage(imageElement);
            
            const inputTensor = new ort.Tensor('float32', imageData.data, imageData.dims);
            
            const feeds = {};
            feeds[this.inputName] = inputTensor;
            
            const results = await this.session.run(feeds);
            
            const outputData = results[this.outputName].data;
            
            const predictions = this.processOutput(outputData);
            
            return predictions;
        } catch (error) {
            console.error('WebAssembly推理失败:', error);
            throw error;
        }
    }

    preprocessImage(imageElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const inputSize = 224;
        canvas.width = inputSize;
        canvas.height = inputSize;
        
        ctx.drawImage(imageElement, 0, 0, inputSize, inputSize);
        
        const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
        const data = imageData.data;
        
        const float32Data = new Float32Array(3 * inputSize * inputSize);
        
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i] / 255.0;
            const g = data[i + 1] / 255.0;
            const b = data[i + 2] / 255.0;
            
            const pixelIndex = i / 4;
            
            float32Data[pixelIndex] = (r - mean[0]) / std[0];
            float32Data[pixelIndex + inputSize * inputSize] = (g - mean[1]) / std[1];
            float32Data[pixelIndex + 2 * inputSize * inputSize] = (b - mean[2]) / std[2];
        }
        
        return {
            data: float32Data,
            dims: [1, 3, inputSize, inputSize]
        };
    }

    processOutput(outputData) {
        const predictions = [];
        
        for (let i = 0; i < outputData.length; i++) {
            predictions.push({
                classIndex: i,
                probability: outputData[i]
            });
        }
        
        predictions.sort((a, b) => b.probability - a.probability);
        
        return predictions;
    }

    async loadClassMapping(mappingPath) {
        try {
            const response = await fetch(mappingPath);
            const mapping = await response.json();
            return mapping.idx_to_class;
        } catch (error) {
            console.error('加载类别映射失败:', error);
            return null;
        }
    }
}

window.WasmInference = WasmInference;
