<#
.SYNOPSIS
    ARDC SkillHub 一键安装脚本
.DESCRIPTION
    为 Windows PowerShell 用户提供一键安装 ARDC SkillHub 的脚本
.NOTES
    Version:        1.0.0
    Author:         ARDC Team
    Creation Date:  2024
    Requires:       PowerShell 5.1+
#>

param(
    [switch]$Force
)

# 颜色定义
$Green = [ConsoleColor]::Green
$Yellow = [ConsoleColor]::Yellow
$Red = [ConsoleColor]::Red
$Blue = [ConsoleColor]::Cyan
$Default = [ConsoleColor]::Gray

# 版本信息
$VERSION = "1.0.0"

# 下载地址
$SKILL_SYNC_URL = "https://47.79.91.89:8888/api/install/ardc-skill-sync.py"
$CONFIG_URL = "https://47.79.91.89:8888/api/install/config.json"

# 安装目录
$INSTALL_DIR = "$env:USERPROFILE\.ardc"
$SKILL_DIR = "$INSTALL_DIR\skills"
$BIN_DIR = "$env:USERPROFILE\.local\bin"

function Write-Color {
    param(
        [string]$Text,
        [ConsoleColor]$Color = $Default
    )
    $original = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Text
    $Host.UI.RawUI.ForegroundColor = $original
}

function Write-Success {
    Write-Color "✓ $args" $Green
}

function Write-Error {
    Write-Color "✗ $args" $Red
}

function Write-Info {
    Write-Color "ℹ $args" $Blue
}

function Write-Warning {
    Write-Color "⚠ $args" $Yellow
}

# ============================================================
# 检查 PowerShell 版本
# ============================================================
function Check-PowerShell {
    Write-Info "检查 PowerShell 版本..."
    
    $psVersion = $PSVersionTable.PSVersion.Major
    if ($psVersion -lt 5) {
        Write-Error "PowerShell 版本过低: v$psVersion"
        Write-Info "需要 PowerShell 5.1+"
        exit 1
    }
    
    Write-Success "PowerShell v$($PSVersionTable.PSVersion)"
}

# ============================================================
# 检查 Python 版本
# ============================================================
function Check-Python {
    Write-Info "检查 Python 环境..."
    
    try {
        $pythonPath = Get-Command python3 -ErrorAction Stop
    } catch {
        try {
            $pythonPath = Get-Command python -ErrorAction Stop
        } catch {
            Write-Error "Python 未安装"
            Write-Info "请安装 Python 3.8+ 后重试"
            exit 1
        }
    }
    
    $pythonVersion = python --version 2>&1 | Select-Object -First 1
    $versionMatch = [regex]::Match($pythonVersion, '(\d+)\.(\d+)')
    
    if ($versionMatch.Success) {
        $major = [int]$versionMatch.Groups[1].Value
        $minor = [int]$versionMatch.Groups[2].Value
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Error "Python 版本过低: $pythonVersion"
            Write-Info "需要 Python 3.8+"
            exit 1
        }
        
        Write-Success "$pythonVersion"
    } else {
        Write-Error "无法获取 Python 版本"
        exit 1
    }
}

# ============================================================
# 创建目录结构
# ============================================================
function Create-Directories {
    Write-Info "创建目录结构..."
    
    if (-not (Test-Path $SKILL_DIR)) {
        New-Item -ItemType Directory -Path $SKILL_DIR -Force | Out-Null
    }
    
    if (-not (Test-Path $BIN_DIR)) {
        New-Item -ItemType Directory -Path $BIN_DIR -Force | Out-Null
    }
    
    Write-Success "目录创建完成"
}

# ============================================================
# 下载技能同步工具
# ============================================================
function Download-SkillSync {
    Write-Info "下载技能同步工具..."
    
    try {
        Invoke-WebRequest -Uri $SKILL_SYNC_URL -OutFile "$BIN_DIR\ardc-skill-sync.py" -UseBasicParsing
        Write-Success "技能同步工具下载完成"
    } catch {
        Write-Error "下载失败: $_"
        exit 1
    }
}

# ============================================================
# 创建启动脚本
# ============================================================
function Create-Launcher {
    Write-Info "创建启动脚本..."
    
    $launcherContent = @"
@echo off
python3 "%USERPROFILE%\.local\bin\ardc-skill-sync.py" %*
"@
    
    $launcherPath = "$BIN_DIR\ardc-skill-sync.bat"
    Set-Content -Path $launcherPath -Value $launcherContent -Encoding UTF8
    
    Write-Success "启动脚本创建完成"
}

# ============================================================
# 创建配置文件
# ============================================================
function Create-Config {
    Write-Info "创建配置文件..."
    
    $configPath = "$INSTALL_DIR\config.json"
    
    if (-not (Test-Path $configPath)) {
        $configContent = @"
{
  "skill_hub_url": "http://47.79.91.89:8888",
  "timeout": 30,
  "log_level": "INFO",
  "auto_update": true
}
"@
        Set-Content -Path $configPath -Value $configContent -Encoding UTF8
        Write-Success "配置文件创建完成"
    } else {
        Write-Success "配置文件已存在"
    }
}

# ============================================================
# 安装依赖
# ============================================================
function Install-Dependencies {
    Write-Info "安装依赖包..."
    
    try {
        python -m pip install requests --quiet
        Write-Success "依赖安装完成"
    } catch {
        Write-Warning "依赖安装失败: $_"
    }
}

# ============================================================
# 设置环境变量
# ============================================================
function Setup-Environment {
    Write-Info "设置环境变量..."
    
    $path = [Environment]::GetEnvironmentVariable("PATH", "User")
    if (-not $path.Contains($BIN_DIR)) {
        $newPath = "$BIN_DIR;$path"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Success "环境变量设置完成"
    } else {
        Write-Success "环境变量已存在"
    }
}

# ============================================================
# 显示安装成功信息
# ============================================================
function Show-Success {
    Write-Host
    Write-Color "==============================================" $Green
    Write-Color "          安装成功！" $Green
    Write-Color "==============================================" $Green
    Write-Host
    
    Write-Color "安装位置:" $Blue
    Write-Host "  - 技能目录: $SKILL_DIR"
    Write-Host "  - 工具脚本: $BIN_DIR\ardc-skill-sync.py"
    Write-Host "  - 配置文件: $INSTALL_DIR\config.json"
    Write-Host
    
    Write-Color "使用方法:" $Blue
    Write-Host "  # 登录认证"
    Write-Host "  ardc-skill-sync login"
    Write-Host
    Write-Host "  # 查看技能列表"
    Write-Host "  ardc-skill-sync list"
    Write-Host
    Write-Host "  # 安装技能"
    Write-Host "  ardc-skill-sync install ardc-collector"
    Write-Host
    Write-Host "  # 检查更新"
    Write-Host "  ardc-skill-sync check"
    Write-Host
    
    Write-Warning "注意: 需要重新打开 PowerShell 以便环境变量生效"
    Write-Host
}

# ============================================================
# 主流程
# ============================================================
function Main {
    Write-Color "==============================================" $Blue
    Write-Color "       ARDC SkillHub 一键安装脚本 v$VERSION" $Blue
    Write-Color "==============================================" $Blue
    Write-Host
    
    Check-PowerShell
    Check-Python
    Create-Directories
    Download-SkillSync
    Create-Launcher
    Create-Config
    Install-Dependencies
    Setup-Environment
    Show-Success
}

# 执行安装
Main
