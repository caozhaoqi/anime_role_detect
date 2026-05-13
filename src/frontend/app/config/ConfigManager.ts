// 配置管理器

interface UIConfig {
  theme: string;
  enableDarkMode: boolean;
  animateTransitions: boolean;
  showPlatformInfo: boolean;
  enableNotifications: boolean;
}

interface FeaturesConfig {
  enableModelSelection: boolean;
  enableCoremlSwitch: boolean;
  enableAttributesSwitch: boolean;
  enableMultiRoleSwitch: boolean;
  enableHistoryPanel: boolean;
  enableDragDrop: boolean;
  enableCopyDownload: boolean;
  enableImagePreview: boolean;
}

interface APIConfig {
  baseUrl: string;
  timeout: number;
  retryCount: number;
  retryDelay: number;
}

interface MessagesConfig {
  welcomeMessage: string;
  processingMessage: string;
  errorMessage: string;
  successMessage: string;
  loginSuccessMessage: string;
  loginErrorMessage: string;
}

interface ValidationConfig {
  maxImageSize: number;
  allowedFormats: string[];
  minImageDimension: number;
}

interface AppearanceConfig {
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
  successColor: string;
  warningColor: string;
  errorColor: string;
  fontFamily: string;
  borderRadius: string;
  shadow: string;
}

interface LayoutConfig {
  sidebarWidth: string;
  headerHeight: string;
  footerHeight: string;
  contentPadding: string;
}

interface AnimationsConfig {
  duration: number;
  easing: string;
  enableHoverEffects: boolean;
  enableLoadingAnimations: boolean;
}

export interface AppConfig {
  ui: UIConfig;
  features: FeaturesConfig;
  api: APIConfig;
  messages: MessagesConfig;
  validation: ValidationConfig;
  appearance: AppearanceConfig;
  layout: LayoutConfig;
  animations: AnimationsConfig;
}

class ConfigManager {
  private static instance: ConfigManager;
  private config: AppConfig;
  private loaded: boolean = false;

  private constructor() {
    this.config = this.getDefaultConfig();
    this.loadConfig();
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  private getDefaultConfig(): AppConfig {
    return {
      ui: {
        theme: "light",
        enableDarkMode: true,
        animateTransitions: true,
        showPlatformInfo: true,
        enableNotifications: true
      },
      features: {
        enableModelSelection: true,
        enableCoremlSwitch: true,
        enableAttributesSwitch: true,
        enableMultiRoleSwitch: true,
        enableHistoryPanel: true,
        enableDragDrop: true,
        enableCopyDownload: true,
        enableImagePreview: true
      },
      api: {
        baseUrl: "/api",
        timeout: 30000,
        retryCount: 3,
        retryDelay: 1000
      },
      messages: {
        welcomeMessage: "你好！我是动漫角色识别助手。请上传一张动漫角色图片，我将尝试识别出这个角色。",
        processingMessage: "正在识别...",
        errorMessage: "识别过程中出现错误，请重试。",
        successMessage: "识别完成！",
        loginSuccessMessage: "登录成功！",
        loginErrorMessage: "登录失败，请检查用户名和密码。"
      },
      validation: {
        maxImageSize: 10485760,
        allowedFormats: ["image/jpeg", "image/png", "image/gif", "image/webp"],
        minImageDimension: 100
      },
      appearance: {
        primaryColor: "#3b82f6",
        secondaryColor: "#8b5cf6",
        accentColor: "#ec4899",
        successColor: "#10b981",
        warningColor: "#f59e0b",
        errorColor: "#ef4444",
        fontFamily: "sans-serif",
        borderRadius: "0.5rem",
        shadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
      },
      layout: {
        sidebarWidth: "280px",
        headerHeight: "6rem",
        footerHeight: "4rem",
        contentPadding: "1.5rem"
      },
      animations: {
        duration: 300,
        easing: "ease-in-out",
        enableHoverEffects: true,
        enableLoadingAnimations: true
      }
    };
  }

  private async loadConfig(): Promise<void> {
    try {
      const response = await fetch('/api/config');
      if (response.ok) {
        const config = await response.json();
        this.config = { ...this.config, ...config };
        this.loaded = true;
      } else {
        console.warn('配置文件加载失败，使用默认配置');
        this.loaded = true;
      }
    } catch (error) {
      console.warn('配置文件加载失败，使用默认配置:', error);
      this.loaded = true;
    }
  }

  public getConfig(): AppConfig {
    return this.config;
  }

  public getUIConfig(): UIConfig {
    return this.config.ui;
  }

  public getFeaturesConfig(): FeaturesConfig {
    return this.config.features;
  }

  public getAPIConfig(): APIConfig {
    return this.config.api;
  }

  public getMessagesConfig(): MessagesConfig {
    return this.config.messages;
  }

  public getValidationConfig(): ValidationConfig {
    return this.config.validation;
  }

  public getAppearanceConfig(): AppearanceConfig {
    return this.config.appearance;
  }

  public getLayoutConfig(): LayoutConfig {
    return this.config.layout;
  }

  public getAnimationsConfig(): AnimationsConfig {
    return this.config.animations;
  }

  public isLoaded(): boolean {
    return this.loaded;
  }

  public updateConfig(newConfig: Partial<AppConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  public updateUIConfig(newUIConfig: Partial<UIConfig>): void {
    this.config.ui = { ...this.config.ui, ...newUIConfig };
  }

  public updateFeaturesConfig(newFeaturesConfig: Partial<FeaturesConfig>): void {
    this.config.features = { ...this.config.features, ...newFeaturesConfig };
  }
}

export default ConfigManager.getInstance();
