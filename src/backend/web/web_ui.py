import gradio as gr
import os
import tempfile
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification
from src.core.exception_handling.exception_handling import ExceptionHandling

class WebUI:
    def __init__(self, index_path="role_index", threshold=0.7):
        """初始化Web UI"""
        self.index_path = index_path
        self.threshold = threshold
        
        # 初始化各个模块
        self.preprocessor = Preprocessing()
        self.extractor = FeatureExtraction()
        self.classifier = Classification(index_path, threshold)
        
        # 尝试初始化标签生成器
        try:
            from src.core.tagging.tagging import Tagging
            self.tagger = Tagging()
        except ImportError:
            self.tagger = None
        
        self.error_handler = ExceptionHandling()
        
        # 添加结果缓存
        self.result_cache = {}
        # 缓存大小限制
        self.cache_size = 100
        
        # 检查索引文件是否存在
        if not os.path.exists(f"{index_path}.faiss") or not os.path.exists(f"{index_path}_mapping.json"):
            print(f"警告: 索引文件不存在: {index_path}.faiss 或 {index_path}_mapping.json")
            print("请先运行 data_preparation.py 构建索引")
    
    def process_image(self, image):
        """处理上传的图片"""
        try:
            # 为图片生成缓存键
            import hashlib
            import io
            
            # 将图片转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # 生成哈希值作为缓存键
            cache_key = hashlib.md5(img_bytes).hexdigest()
            
            # 检查缓存
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
            
            # 将PIL Image保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file, format="JPEG")
                temp_image_path = temp_file.name
            
            # 预处理图像
            normalized_img, boxes = self.preprocessor.process(temp_image_path)
            
            # 提取特征
            feature = self.extractor.extract_features(normalized_img)
            
            # 分类
            role, similarity = self.classifier.classify(feature)
            
            # 获取标签
            tags = []
            if self.tagger:
                tags = self.tagger.get_tags(temp_image_path)
            
            # 清理临时文件
            os.unlink(temp_image_path)
            
            # 生成结果
            if role == "unknown" or similarity < self.threshold:
                result = f"无法识别该角色（相似度: {similarity:.4f}）"
            else:
                result = f"识别结果: {role}（相似度: {similarity:.4f}）"
            
            tags_str = ", ".join(tags) if tags else "无"
            
            # 构建结果
            result_tuple = (result, tags_str, normalized_img)
            
            # 更新缓存
            if len(self.result_cache) >= self.cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            self.result_cache[cache_key] = result_tuple
            
            return result_tuple
        except Exception as e:
            # 处理异常
            error_message = str(e)
            self.error_handler.handle_exception(e, "Web UI图片处理")
            return f"处理失败: {error_message}", "无", image
    
    def process_multiple_characters(self, image):
        """处理图片中的多个角色"""
        try:
            # 为图片生成缓存键
            import hashlib
            import io
            
            # 将图片转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # 生成哈希值作为缓存键
            cache_key = hashlib.md5(img_bytes).hexdigest()
            
            # 检查缓存
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
            
            # 将PIL Image保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file, format="JPEG")
                temp_image_path = temp_file.name
            
            # 处理多个角色
            processed_characters = self.preprocessor.process_multiple_characters(temp_image_path)
            
            if not processed_characters:
                result_tuple = ("未检测到角色", "无", image)
                # 更新缓存
                if len(self.result_cache) >= self.cache_size:
                    oldest_key = next(iter(self.result_cache))
                    del self.result_cache[oldest_key]
                self.result_cache[cache_key] = result_tuple
                return result_tuple
            
            # 提取特征
            characters_with_features = self.extractor.extract_features_from_multiple_characters(processed_characters)
            
            # 分类
            classified_characters = self.classifier.classify_multiple_characters(characters_with_features)
            
            # 清理临时文件
            os.unlink(temp_image_path)
            
            # 生成结果
            results = []
            for i, char in enumerate(classified_characters):
                role = char.get('role', 'unknown')
                similarity = char.get('similarity', 0.0)
                confidence = char.get('confidence', 0.0)
                
                if role == "unknown" or similarity < self.threshold:
                    result = f"角色{i+1}: 无法识别（检测置信度: {confidence:.4f}，相似度: {similarity:.4f}）"
                else:
                    result = f"角色{i+1}: {role}（检测置信度: {confidence:.4f}，相似度: {similarity:.4f}）"
                results.append(result)
            
            result_str = "\n".join(results)
            tags_str = f"检测到 {len(classified_characters)} 个角色"
            
            # 构建结果
            result_tuple = (result_str, tags_str, image)
            
            # 更新缓存
            if len(self.result_cache) >= self.cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.result_cache))
                del self.result_cache[oldest_key]
            self.result_cache[cache_key] = result_tuple
            
            return result_tuple
        except Exception as e:
            # 处理异常
            error_message = str(e)
            self.error_handler.handle_exception(e, "Web UI多角色处理")
            return f"处理失败: {error_message}", "无", image
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="二次元角色识别与分类系统", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 二次元角色识别与分类系统
            上传一张二次元角色图片，系统将自动识别角色并添加标签。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="上传图片")
                    with gr.Row():
                        submit_btn = gr.Button("单角色识别")
                        multi_submit_btn = gr.Button("多角色识别")
                
                with gr.Column(scale=2):
                    output_result = gr.Textbox(label="识别结果", lines=4)
                    output_tags = gr.Textbox(label="标签", lines=2)
                    output_image = gr.Image(label="处理后图片")
            
            # 设置按钮点击事件
            submit_btn.click(
                fn=self.process_image,
                inputs=[input_image],
                outputs=[output_result, output_tags, output_image]
            )
            
            # 设置多角色识别按钮点击事件
            multi_submit_btn.click(
                fn=self.process_multiple_characters,
                inputs=[input_image],
                outputs=[output_result, output_tags, output_image]
            )
            
            # 设置图片上传自动处理
            input_image.change(
                fn=self.process_image,
                inputs=[input_image],
                outputs=[output_result, output_tags, output_image]
            )
            
            # 添加增量学习功能
            with gr.Row():
                gr.Markdown("""
                ## 修正识别结果
                如果识别结果不正确，请输入正确的角色名称并点击「更新索引」按钮
                """)
            
            with gr.Row():
                correct_role_input = gr.Textbox(label="正确的角色名称")
                update_index_btn = gr.Button("更新索引")
            
            update_index_output = gr.Textbox(label="更新结果", lines=2)
            
            def update_index(image, correct_role):
                """更新索引"""
                if not image:
                    return "请先上传图片"
                
                if not correct_role:
                    return "请输入正确的角色名称"
                
                try:
                    # 将PIL Image保存为临时文件
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        image.save(temp_file, format="JPEG")
                        temp_image_path = temp_file.name
                    
                    # 执行增量学习
                    success = self.classifier.incremental_learning(temp_image_path, correct_role)
                    
                    # 清理临时文件
                    os.unlink(temp_image_path)
                    
                    if success:
                        # 重新保存索引
                        self.classifier.save_index(self.index_path)
                        return f"索引更新成功！已将图片添加到角色 '{correct_role}' 的索引中"
                    else:
                        return "索引更新失败，请查看控制台输出"
                except Exception as e:
                    return f"更新失败: {str(e)}"
            
            update_index_btn.click(
                fn=update_index,
                inputs=[input_image, correct_role_input],
                outputs=[update_index_output]
            )
            
            # 添加关于信息
            gr.Markdown("""
            ---  
            **系统说明**
            - 使用CLIP模型进行特征提取
            - 使用Faiss进行向量检索
            - 使用DeepDanbooru进行标签生成
            - 相似度阈值: 0.7
            - 支持增量学习，可手动修正错误分类
            """)
        
        return interface
    
    def launch(self, share=False, server_port=7860):
        """启动Web UI"""
        interface = self.create_interface()
        interface.launch(share=share, server_port=server_port)

if __name__ == "__main__":
    # 创建Web UI实例并启动
    web_ui = WebUI()
    web_ui.launch(share=False)
