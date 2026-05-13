import { NextResponse } from 'next/server';

export async function GET() {
  const config = {
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
      maxImageSize: 10 * 1024 * 1024,
      allowedFormats: ["jpg", "jpeg", "png", "webp"],
      minImageDimension: 64
    },
    appearance: {
      primaryColor: "#3b82f6",
      secondaryColor: "#6366f1",
      accentColor: "#8b5cf6",
      successColor: "#10b981",
      warningColor: "#f59e0b",
      errorColor: "#ef4444",
      fontFamily: "system-ui, -apple-system, sans-serif",
      borderRadius: "0.5rem",
      shadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)"
    },
    layout: {
      sidebarWidth: "16rem",
      headerHeight: "4rem",
      footerHeight: "3rem",
      contentPadding: "1.5rem"
    },
    animations: {
      duration: 300,
      easing: "ease-in-out",
      enableHoverEffects: true,
      enableLoadingAnimations: true
    }
  };

  return NextResponse.json(config);
}