<template>
  <div class="max-w-4xl mx-auto">
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-2">使用指南</h1>
      <p class="text-gray-600">学习如何安装和使用 ARDC SkillHub 的 CLI 工具</p>
    </div>
    
    <!-- 快速开始 -->
    <section class="bg-gradient-to-r from-primary-500 to-primary-600 rounded-2xl p-6 mb-8 text-white">
      <h2 class="text-xl font-bold mb-4 flex items-center gap-2">
        <Zap class="w-5 h-5" />
        快速开始
      </h2>
      <div class="grid md:grid-cols-3 gap-4">
        <div class="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
          <div class="text-2xl font-bold mb-1">1</div>
          <div>安装 CLI 工具</div>
        </div>
        <div class="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
          <div class="text-2xl font-bold mb-1">2</div>
          <div>登录认证</div>
        </div>
        <div class="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
          <div class="text-2xl font-bold mb-1">3</div>
          <div>安装技能</div>
        </div>
      </div>
    </section>
    
    <!-- 安装 CLI -->
    <section class="bg-white rounded-2xl shadow-lg p-6 mb-8">
      <h2 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Terminal class="w-5 h-5 text-primary-500" />
        安装 CLI 工具
      </h2>
      
      <div class="grid md:grid-cols-2 gap-6">
        <!-- macOS / Linux -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <Laptop class="w-4 h-4" />
            macOS / Linux
          </h3>
          <div class="bg-gray-900 rounded-xl p-4 overflow-x-auto relative">
            <button
              class="copy-btn absolute top-2 right-2 p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
              :class="copiedCmd === 'macos' ? 'bg-green-700 text-green-400' : 'text-gray-400 hover:text-gray-300'"
              @click="copyToClipboard('curl -fsSL http://47.79.91.89:8888/api/install/install.sh | sh', 'cmd', 'macos')"
              title="复制命令"
            >
              <Check v-if="copiedCmd === 'macos'" class="w-4 h-4" />
              <Copy v-else class="w-4 h-4" />
            </button>
            <pre class="text-green-400 text-sm"><code>curl -fsSL http://47.79.91.89:8888/api/install/install.sh | sh</code></pre>
          </div>
          <p class="text-sm text-gray-500 mt-3">需要 Python 3.8+ 环境</p>
        </div>
        
        <!-- Windows -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <Monitor class="w-4 h-4" />
            Windows PowerShell
          </h3>
          <div class="bg-gray-900 rounded-xl p-4 overflow-x-auto relative">
            <button
              class="copy-btn absolute top-2 right-2 p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
              :class="copiedCmd === 'windows' ? 'bg-green-700 text-green-400' : 'text-gray-400 hover:text-gray-300'"
              @click="copyToClipboard('irm http://47.79.91.89:8888/api/install/install.ps1 | iex', 'cmd', 'windows')"
              title="复制命令"
            >
              <Check v-if="copiedCmd === 'windows'" class="w-4 h-4" />
              <Copy v-else class="w-4 h-4" />
            </button>
            <pre class="text-green-400 text-sm"><code>irm http://47.79.91.89:8888/api/install/install.ps1 | iex</code></pre>
          </div>
          <p class="text-sm text-gray-500 mt-3">以管理员身份运行 PowerShell</p>
        </div>
      </div>
    </section>
    
    <!-- CLI 命令 -->
    <section class="bg-white rounded-2xl shadow-lg p-6 mb-8">
      <h2 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Command class="w-5 h-5 text-primary-500" />
        CLI 命令说明
      </h2>
      
      <div class="space-y-4">
        <div v-for="cmd in commands" :key="cmd.name" class="border border-gray-100 rounded-xl p-4 hover:bg-gray-50 transition-colors">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-3">
              <code class="font-mono text-primary-600">{{ cmd.name }}</code>
              <button
                class="copy-btn p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                :class="copiedCmd === cmd.name ? 'bg-green-100 text-green-600' : 'text-gray-400 hover:text-gray-600'"
                @click="copyToClipboard(cmd.name, 'cmd', cmd.name)"
                title="复制命令"
              >
                <Check v-if="copiedCmd === cmd.name" class="w-4 h-4" />
                <Copy v-else class="w-4 h-4" />
              </button>
            </div>
            <span class="text-sm text-gray-500">{{ cmd.description }}</span>
          </div>
          <div v-if="cmd.example" class="bg-gray-50 rounded-lg p-3">
            <div class="flex items-center justify-between mb-1">
              <p class="text-xs text-gray-400">示例:</p>
              <button
                class="copy-btn p-1 rounded hover:bg-gray-200 transition-colors"
                :class="copiedExample === cmd.name ? 'text-green-600' : 'text-gray-400 hover:text-gray-600'"
                @click="copyToClipboard(cmd.example, 'example', cmd.name)"
                title="复制示例"
              >
                <Check v-if="copiedExample === cmd.name" class="w-3 h-3" />
                <Copy v-else class="w-3 h-3" />
              </button>
            </div>
            <code class="text-sm text-gray-700">{{ cmd.example }}</code>
          </div>
        </div>
      </div>
    </section>
    
    <!-- 使用流程 -->
    <section class="bg-white rounded-2xl shadow-lg p-6 mb-8">
      <h2 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <GitBranch class="w-5 h-5 text-primary-500" />
        使用流程
      </h2>
      
      <div class="space-y-6">
        <div v-for="(step, index) in steps" :key="index" class="flex gap-4">
          <div class="flex-shrink-0 w-10 h-10 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center font-bold">
            {{ index + 1 }}
          </div>
          <div class="flex-1">
            <h3 class="font-medium text-gray-900 mb-1">{{ step.title }}</h3>
            <p class="text-sm text-gray-600">{{ step.description }}</p>
            <div v-if="step.code" class="mt-3 bg-gray-50 rounded-lg p-3">
              <code class="text-sm text-gray-700">{{ step.code }}</code>
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- 技能开发者指南 -->
    <section class="bg-gradient-to-r from-purple-50 to-blue-50 rounded-2xl shadow-lg p-6 mb-8">
      <h2 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Terminal class="w-5 h-5 text-purple-500" />
        技能开发者指南
      </h2>
      
      <div class="space-y-6">
        <!-- 版本更新流程 -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">📦 版本更新流程</h3>
          <div class="bg-white rounded-xl p-4 space-y-3">
            <div class="flex items-start gap-3">
              <span class="text-primary-600 font-mono text-sm">1.</span>
              <div>
                <p class="font-medium text-gray-900">开发新版本</p>
                <p class="text-sm text-gray-600">在本地开发并测试技能的新功能</p>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <span class="text-primary-600 font-mono text-sm">2.</span>
              <div>
                <p class="font-medium text-gray-900">更新版本号</p>
                <p class="text-sm text-gray-600">遵循 Semantic Versioning（语义化版本）规范</p>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <span class="text-primary-600 font-mono text-sm">3.</span>
              <div>
                <p class="font-medium text-gray-900">发布新版本</p>
                <p class="text-sm text-gray-600">使用 API 或 CLI 工具发布新版本</p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 版本规范 -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">📝 版本号规范</h3>
          <div class="bg-white rounded-xl p-4">
            <pre class="text-sm text-gray-700 font-mono bg-gray-50 rounded-lg p-3">MAJOR.MINOR.PATCH
├── MAJOR: 重大变更，不兼容的 API 修改
├── MINOR: 新功能，向后兼容
└── PATCH: 修复 bug，向后兼容</pre>
          </div>
        </div>
        
        <!-- 发布命令 -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">🚀 发布新版本命令</h3>
          <div class="bg-gray-900 rounded-xl p-4 overflow-x-auto relative">
            <button
              class="copy-btn absolute top-2 right-2 p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
              :class="copiedCmd === 'publish' ? 'bg-green-700 text-green-400' : 'text-gray-400 hover:text-gray-300'"
              @click="copyToClipboard('ardc-skill-sync publish --skill ardc-collector --version 1.1.0 --changelog 新增功能描述', 'cmd', 'publish')"
              title="复制命令"
            >
              <Check v-if="copiedCmd === 'publish'" class="w-4 h-4" />
              <Copy v-else class="w-4 h-4" />
            </button>
            <pre class="text-green-400 text-sm"><code>ardc-skill-sync publish \
  --skill ardc-collector \
  --version 1.1.0 \
  --changelog "新增功能描述"</code></pre>
          </div>
        </div>
        
        <!-- API 接口 -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">🔌 API 接口</h3>
          <div class="space-y-3">
            <div class="bg-white rounded-xl p-4">
              <div class="flex items-center gap-2 mb-2">
                <span class="px-2 py-1 bg-green-100 text-green-600 text-xs font-medium rounded">POST</span>
                <code class="text-sm text-gray-700">/api/skills/{skill_name}/versions</code>
              </div>
              <p class="text-sm text-gray-600">发布新版本，需要 Token 认证</p>
            </div>
            <div class="bg-white rounded-xl p-4">
              <div class="flex items-center gap-2 mb-2">
                <span class="px-2 py-1 bg-blue-100 text-blue-600 text-xs font-medium rounded">GET</span>
                <code class="text-sm text-gray-700">/api/skills/{skill_name}/versions</code>
              </div>
              <p class="text-sm text-gray-600">查看技能的所有版本历史</p>
            </div>
            <div class="bg-white rounded-xl p-4">
              <div class="flex items-center gap-2 mb-2">
                <span class="px-2 py-1 bg-blue-100 text-blue-600 text-xs font-medium rounded">GET</span>
                <code class="text-sm text-gray-700">/api/skills/{skill_name}/check-update?current_version=1.0.0</code>
              </div>
              <p class="text-sm text-gray-600">检查是否有新版本可用</p>
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <!-- 常见问题 -->
    <section class="bg-white rounded-2xl shadow-lg p-6">
      <h2 class="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <HelpCircle class="w-5 h-5 text-primary-500" />
        常见问题
      </h2>
      
      <div class="space-y-4">
        <div v-for="faq in faqs" :key="faq.question" class="border border-gray-100 rounded-xl overflow-hidden">
          <button 
            class="w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors flex items-center justify-between"
            @click="toggleFaq(faq.question)"
          >
            <span class="font-medium text-gray-900">{{ faq.question }}</span>
            <ChevronDown 
              class="w-5 h-5 text-gray-400 transition-transform"
              :class="{ 'rotate-180': openFaqs.includes(faq.question) }"
            />
          </button>
          <div v-if="openFaqs.includes(faq.question)" class="px-4 pb-4">
            <p class="text-sm text-gray-600">{{ faq.answer }}</p>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { 
  Zap, Terminal, Monitor, Laptop, Command, 
  GitBranch, HelpCircle, ChevronDown, Copy, Check 
} from 'lucide-vue-next'

const openFaqs = ref([])
const copiedCmd = ref(null)
const copiedExample = ref(null)

const toggleFaq = (question) => {
  const index = openFaqs.value.indexOf(question)
  if (index > -1) {
    openFaqs.value.splice(index, 1)
  } else {
    openFaqs.value.push(question)
  }
}

const copyToClipboard = async (text, type, cmdName) => {
  try {
    await navigator.clipboard.writeText(text)
    if (type === 'cmd') {
      copiedCmd.value = cmdName
      setTimeout(() => { copiedCmd.value = null }, 2000)
    } else {
      copiedExample.value = cmdName
      setTimeout(() => { copiedExample.value = null }, 2000)
    }
  } catch (err) {
    console.error('复制失败:', err)
  }
}

const commands = [
  {
    name: 'ardc-skill-sync login',
    description: '用户名密码登录认证',
    example: 'ardc-skill-sync login'
  },
  {
    name: 'ardc-skill-sync register',
    description: '注册新账户',
    example: 'ardc-skill-sync register'
  },
  {
    name: 'ardc-skill-sync status',
    description: '显示本地配置与检测到的技能目录'
  },
  {
    name: 'ardc-skill-sync check',
    description: '检查已安装技能的更新情况'
  },
  {
    name: 'ardc-skill-sync sync',
    description: '同步更新所有已安装技能'
  },
  {
    name: 'ardc-skill-sync list',
    description: '查询 SkillHub 上所有已发布的技能',
    example: 'ardc-skill-sync list'
  },
  {
    name: 'ardc-skill-sync install <skill-name>',
    description: '安装指定技能',
    example: 'ardc-skill-sync install ardc-collector'
  },
  {
    name: 'ardc-skill-sync uninstall <skill-name>',
    description: '卸载指定技能',
    example: 'ardc-skill-sync uninstall ardc-collector'
  },
  {
    name: 'ardc-skill-sync help',
    description: '显示帮助信息'
  }
]

const steps = [
  {
    title: '安装 CLI 工具',
    description: '使用一键安装脚本安装 ARDC SkillHub CLI 工具',
    code: 'curl -fsSL http://47.79.91.89:8888/api/install/install.sh | sh'
  },
  {
    title: '登录认证',
    description: '使用注册的账户登录，获取访问令牌',
    code: 'ardc-skill-sync login'
  },
  {
    title: '浏览技能',
    description: '查看 SkillHub 上可用的所有技能',
    code: 'ardc-skill-sync list'
  },
  {
    title: '安装技能',
    description: '安装您需要的技能，工具会自动处理依赖',
    code: 'ardc-skill-sync install ardc-collector'
  },
  {
    title: '使用技能',
    description: '技能安装完成后，即可在项目中使用',
    code: 'python -m ardc.collector --help'
  }
]

const faqs = [
  {
    question: '安装时提示权限不足怎么办？',
    answer: '在 macOS/Linux 上可以尝试使用 sudo 运行安装脚本，或在 Windows 上以管理员身份运行 PowerShell。'
  },
  {
    question: '登录失败怎么办？',
    answer: '请检查用户名和密码是否正确。如果忘记密码，可以重新注册一个新账户。也可以使用离线模式继续使用已安装的技能。'
  },
  {
    question: '技能安装失败怎么办？',
    answer: '请检查网络连接是否正常，以及 Python 环境是否满足技能要求。可以使用 `ardc-skill-sync status` 检查本地配置。'
  },
  {
    question: '如何更新技能？',
    answer: '使用 `ardc-skill-sync check` 检查更新，使用 `ardc-skill-sync sync` 同步更新所有技能，或使用 `ardc-skill-sync install <skill-name>` 重新安装指定技能。'
  },
  {
    question: '可以离线使用吗？',
    answer: '是的，CLI 工具支持离线模式。如果无法连接到服务器，工具会自动切换到离线模式，可以继续使用已安装的技能。'
  }
]
</script>
