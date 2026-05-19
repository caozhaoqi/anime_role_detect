<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
    <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="$emit('close')"></div>
    
    <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden animate-slide-up">
      <div class="bg-gradient-to-r from-primary-500 to-primary-600 px-6 py-5">
        <div class="flex items-start justify-between">
          <div>
            <h2 class="text-xl font-bold text-white">{{ skill.name }}</h2>
            <p class="text-primary-100 text-sm mt-1">{{ skill.name }}</p>
          </div>
          <button
            class="p-2 rounded-lg hover:bg-white/10 transition-colors text-white"
            @click="$emit('close')"
          >
            <X class="w-5 h-5" />
          </button>
        </div>
      </div>
      
      <div class="p-6 overflow-y-auto max-h-[calc(90vh-140px)] scrollbar-thin">
        <div class="flex flex-wrap items-center gap-2 mb-4">
          <span :class="['category-badge', `category-${skill.category}`]">
            {{ getCategoryLabel(skill.category) }}
          </span>
          <span :class="['status-tag', `status-${skill.status}`]">
            {{ getStatusLabel(skill.status) }}
          </span>
          <span class="tag tag-category">{{ skill.version }}</span>
        </div>
        
        <div class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <FileText class="w-4 h-4 text-gray-500" />
            描述
          </h3>
          <p class="text-gray-600 text-sm">{{ skill.description || '暂无描述' }}</p>
        </div>
        
        <div class="grid grid-cols-2 gap-4 mb-6">
          <div class="bg-gray-50 rounded-xl p-4">
            <div class="flex items-center gap-2 text-gray-500 text-sm mb-1">
              <User class="w-4 h-4" />
              作者
            </div>
            <p class="font-medium text-gray-900">{{ skill.author }}</p>
          </div>
          <div class="bg-gray-50 rounded-xl p-4">
            <div class="flex items-center gap-2 text-gray-500 text-sm mb-1">
              <Tag class="w-4 h-4" />
              版本
            </div>
            <p class="font-medium text-gray-900">{{ skill.version }}</p>
          </div>
        </div>
        
        <div v-if="skill.tags && skill.tags.length > 0" class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Tag class="w-4 h-4 text-gray-500" />
            标签
          </h3>
          <div class="flex flex-wrap gap-2">
            <span v-for="tag in skill.tags" :key="tag" class="tag tag-category">
              {{ tag }}
            </span>
          </div>
        </div>
        
        <div v-if="skill.dependencies && skill.dependencies.length > 0" class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <GitBranch class="w-4 h-4 text-gray-500" />
            依赖
          </h3>
          <div class="flex flex-wrap gap-2">
            <span v-for="dep in skill.dependencies" :key="dep" class="tag tag-category">
              {{ dep }}
            </span>
          </div>
        </div>
        
        <div v-if="skill.config_schema && skill.config_schema.length > 0" class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Settings class="w-4 h-4 text-gray-500" />
            配置参数
          </h3>
          <div class="space-y-2">
            <div v-for="config in skill.config_schema" :key="config.name" class="bg-gray-50 rounded-lg px-4 py-2">
              <div class="flex items-center justify-between mb-1">
                <span class="text-sm font-medium text-gray-900">{{ config.name }}</span>
                <span class="text-xs text-gray-500">{{ config.type }}</span>
              </div>
              <p class="text-xs text-gray-500">{{ config.description }}</p>
            </div>
          </div>
        </div>
        
        <div class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Server class="w-4 h-4 text-gray-500" />
            运行要求
          </h3>
          <div class="grid grid-cols-3 gap-3">
            <div class="bg-gray-50 rounded-xl p-3 text-center">
              <div class="text-sm text-gray-500 mb-1">内存</div>
              <div class="font-medium text-gray-900">{{ skill.memory_mb }} MB</div>
            </div>
            <div class="bg-gray-50 rounded-xl p-3 text-center">
              <div class="text-sm text-gray-500 mb-1">CPU</div>
              <div class="font-medium text-gray-900">{{ skill.cpu_cores }} 核</div>
            </div>
            <div class="bg-gray-50 rounded-xl p-3 text-center">
              <div class="text-sm text-gray-500 mb-1">运行时</div>
              <div class="font-medium text-gray-900">{{ skill.runtime }}</div>
            </div>
          </div>
        </div>
        
        <div class="mb-6">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <Calendar class="w-4 h-4 text-gray-500" />
            更新时间
          </h3>
          <p class="text-sm text-gray-600">
            创建于 {{ formatDate(skill.created_at) }}
            <span v-if="skill.created_at !== skill.updated_at">
              | 更新于 {{ formatDate(skill.updated_at) }}
            </span>
          </p>
        </div>
        
        <div class="mb-6 bg-blue-50 rounded-xl p-4">
          <h3 class="font-medium text-gray-900 mb-2 flex items-center gap-2">
            <BookOpen class="w-4 h-4 text-blue-500" />
            安装指导
          </h3>
          <div class="text-sm text-gray-600 space-y-2">
            <p>1. 点击下方"安装技能"按钮开始安装</p>
            <p>2. 安装完成后，技能将自动下载并配置</p>
            <p>3. 在命令行使用 <code class="bg-white px-1.5 py-0.5 rounded text-blue-600">ardc-skill-sync list</code> 查看已安装技能</p>
            <p>4. 使用 <code class="bg-white px-1.5 py-0.5 rounded text-blue-600">ardc-skill-sync install {{ skill.name }}</code> 命令手动安装</p>
          </div>
        </div>
        
        <!-- 版本更新提示 -->
        <div v-if="updateInfo && updateInfo.has_update" class="mb-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
          <div class="flex items-start gap-3">
            <AlertCircle class="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 class="font-medium text-green-800 mb-1">发现新版本</h3>
              <p class="text-sm text-green-700">当前版本: <span class="font-medium">{{ updateInfo.current_version }}</span> → 最新版本: <span class="font-bold">{{ updateInfo.latest_version }}</span></p>
              <p class="text-sm text-green-600 mt-1">更新内容: {{ updateInfo.changelog }}</p>
            </div>
          </div>
        </div>
        
        <!-- 版本历史 -->
        <div v-if="versions && versions.length > 0" class="mb-6">
          <div class="flex items-center justify-between mb-3">
            <h3 class="font-medium text-gray-900 flex items-center gap-2">
              <Clock class="w-4 h-4 text-gray-500" />
              版本历史
            </h3>
            <button 
              v-if="skill.installed"
              class="text-sm text-primary-600 hover:text-primary-700 flex items-center gap-1"
              @click="checkUpdate"
              :disabled="checkingUpdate"
            >
              <RefreshCw class="w-3 h-3" :class="{ 'animate-spin': checkingUpdate }" />
              {{ checkingUpdate ? '检查中...' : '检查更新' }}
            </button>
          </div>
          <div class="space-y-2">
            <div 
              v-for="(version, index) in versions" 
              :key="version.version"
              :class="[
                'flex items-start gap-3 p-3 rounded-lg',
                index === 0 ? 'bg-primary-50 border border-primary-100' : 'bg-gray-50'
              ]"
            >
              <div class="flex-shrink-0">
                <span :class="[
                  'px-2 py-1 rounded-full text-xs font-medium',
                  index === 0 ? 'bg-primary-100 text-primary-700' : 'bg-gray-200 text-gray-700'
                ]">
                  {{ version.version }}
                </span>
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-sm text-gray-600 truncate">{{ version.changelog }}</p>
                <p class="text-xs text-gray-400 mt-1">{{ version.release_date }}</p>
              </div>
              <span v-if="index === 0" class="text-xs text-primary-600 font-medium">最新</span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="border-t border-gray-100 px-6 py-4 flex items-center justify-between">
        <button class="btn btn-secondary" @click="$emit('close')">
          <ArrowLeft class="w-4 h-4" />
          返回
        </button>
        
        <div class="flex items-center gap-3">
          <button
            v-if="skill.installed && updateInfo && updateInfo.has_update"
            class="btn btn-lg btn-primary flex items-center gap-2"
            @click="emit('update', skill.name)"
          >
            <RefreshCw class="w-5 h-5" />
            更新技能 ({{ updateInfo.latest_version }})
          </button>
          <button
            :class="[
              'btn btn-lg flex items-center gap-2',
              skill.installed ? 'btn-secondary' : 'btn-primary'
            ]"
            @click="handleAction"
          >
            <component :is="skill.installed ? CheckCircle : Download" class="w-5 h-5" />
            {{ skill.installed ? '已安装' : '安装技能' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {
  X, FileText, User, GitBranch, Tag,
  Settings, Server, Calendar, ArrowLeft, Download, CheckCircle, BookOpen,
  RefreshCw, AlertCircle, Clock
} from 'lucide-vue-next'

import { ref, onMounted } from 'vue'

const props = defineProps({
  skill: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close', 'install', 'uninstall', 'update'])

const versions = ref([])
const updateInfo = ref(null)
const checkingUpdate = ref(false)

const loadVersions = async () => {
  try {
    const response = await fetch(`/api/skills/${props.skill.name}/versions`)
    versions.value = await response.json()
  } catch (error) {
    console.error('Failed to load versions:', error)
  }
}

const checkUpdate = async () => {
  checkingUpdate.value = true
  try {
    const response = await fetch(`/api/skills/${props.skill.name}/check-update?current_version=${props.skill.version}`)
    updateInfo.value = await response.json()
  } catch (error) {
    console.error('Failed to check update:', error)
  } finally {
    checkingUpdate.value = false
  }
}

onMounted(() => {
  loadVersions()
  if (props.skill.installed) {
    checkUpdate()
  }
})

const getCategoryLabel = (name) => {
  const labels = {
    collector: '数据采集',
    cleaner: '数据清洗',
    classifier: '分类识别',
    trainer: '模型训练',
    search: '搜索检索',
    analyzer: '数据分析',
    utility: '工具辅助'
  }
  return labels[name] || name
}

const getStatusLabel = (status) => {
  const labels = {
    stable: '稳定版',
    testing: '测试中',
    development: '开发中',
    deprecated: '已弃用'
  }
  return labels[status] || status
}

const formatDate = (dateStr) => {
  if (!dateStr) return '未知'
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const handleAction = () => {
  if (props.skill.installed) {
    emit('uninstall', props.skill.name)
  } else {
    emit('install', props.skill.name)
  }
}
</script>

<style scoped>
.status-tag {
  @apply px-2 py-1 rounded-full text-xs font-medium;
}

.status-stable { @apply bg-green-100 text-green-700; }
.status-testing { @apply bg-yellow-100 text-yellow-700; }
.status-development { @apply bg-blue-100 text-blue-700; }
.status-deprecated { @apply bg-red-100 text-red-700; }
</style>
