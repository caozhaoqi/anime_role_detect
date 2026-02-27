/**
 * 用户体验数据收集模块
 * 收集推理性能数据、设备信息和用户反馈
 */

class UserExperienceTracker {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.startTime = Date.now();
        this.inferenceData = [];
        this.deviceInfo = this.collectDeviceInfo();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    collectDeviceInfo() {
        return {
            userAgent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            screenResolution: `${window.screen.width}x${window.screen.height}`,
            viewportSize: `${window.innerWidth}x${window.innerHeight}`,
            devicePixelRatio: window.devicePixelRatio,
            touchSupport: 'ontouchstart' in window,
            webGLSupport: this.checkWebGLSupport(),
            webAssemblySupport: typeof WebAssembly !== 'undefined',
            gpuSupport: navigator.gpu !== undefined,
            isApple: /Macintosh|iPhone|iPad|iPod/i.test(navigator.userAgent),
            isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),
            isWindows: /Windows/i.test(navigator.userAgent),
            isLinux: /Linux/i.test(navigator.userAgent)
        };
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
            return false;
        }
    }

    recordInference(inferenceData) {
        const record = {
            sessionId: this.sessionId,
            timestamp: Date.now(),
            inferenceMode: inferenceData.mode,
            inferenceTime: inferenceData.time,
            modelName: inferenceData.modelName,
            imageSize: inferenceData.imageSize,
            success: inferenceData.success,
            error: inferenceData.error || null,
            deviceInfo: this.deviceInfo
        };

        this.inferenceData.push(record);

        // 保存到本地存储
        this.saveToLocalStorage(record);

        // 发送到服务器（如果可用）
        this.sendToServer(record);
    }

    saveToLocalStorage(record) {
        try {
            const key = 'inference_records';
            let records = JSON.parse(localStorage.getItem(key) || '[]');
            records.push(record);

            // 只保留最近的100条记录
            if (records.length > 100) {
                records = records.slice(-100);
            }

            localStorage.setItem(key, JSON.stringify(records));
        } catch (e) {
            console.error('保存到本地存储失败:', e);
        }
    }

    async sendToServer(record) {
        try {
            const response = await fetch('/api/track_inference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(record)
            });

            if (response.ok) {
                console.log('用户体验数据已发送到服务器');
            }
        } catch (e) {
            console.error('发送到服务器失败:', e);
        }
    }

    recordUserFeedback(feedback) {
        const record = {
            sessionId: this.sessionId,
            timestamp: Date.now(),
            feedback: feedback,
            deviceInfo: this.deviceInfo
        };

        try {
            const key = 'user_feedback';
            let records = JSON.parse(localStorage.getItem(key) || '[]');
            records.push(record);
            localStorage.setItem(key, JSON.stringify(records));
        } catch (e) {
            console.error('保存用户反馈失败:', e);
        }
    }

    getStatistics() {
        const records = JSON.parse(localStorage.getItem('inference_records') || '[]');

        if (records.length === 0) {
            return null;
        }

        const stats = {
            totalInferences: records.length,
            byMode: {},
            averageTimes: {},
            successRate: 0,
            deviceTypes: {}
        };

        let successCount = 0;

        records.forEach(record => {
            const mode = record.inferenceMode;
            const time = record.inferenceTime;
            const deviceType = record.deviceInfo.isApple ? 'Apple' :
                              record.deviceInfo.isMobile ? 'Mobile' :
                              record.deviceInfo.isWindows ? 'Windows' :
                              record.deviceInfo.isLinux ? 'Linux' : 'Other';

            // 按模式统计
            if (!stats.byMode[mode]) {
                stats.byMode[mode] = 0;
            }
            stats.byMode[mode]++;

            // 按设备类型统计
            if (!stats.deviceTypes[deviceType]) {
                stats.deviceTypes[deviceType] = 0;
            }
            stats.deviceTypes[deviceType]++;

            // 计算平均时间
            if (!stats.averageTimes[mode]) {
                stats.averageTimes[mode] = { total: 0, count: 0 };
            }
            stats.averageTimes[mode].total += time;
            stats.averageTimes[mode].count++;

            // 成功率
            if (record.success) {
                successCount++;
            }
        });

        // 计算平均时间
        Object.keys(stats.averageTimes).forEach(mode => {
            const data = stats.averageTimes[mode];
            stats.averageTimes[mode] = data.total / data.count;
        });

        stats.successRate = (successCount / records.length) * 100;

        return stats;
    }

    clearOldData() {
        try {
            localStorage.removeItem('inference_records');
            localStorage.removeItem('user_feedback');
            console.log('旧数据已清除');
        } catch (e) {
            console.error('清除数据失败:', e);
        }
    }
}

window.UserExperienceTracker = UserExperienceTracker;
