<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
    <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="$emit('close')"></div>
    
    <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden animate-slide-up">
      <div class="bg-gradient-to-r from-primary-500 to-primary-600 px-6 py-5">
        <div class="flex items-start justify-between">
          <div>
            <h2 class="text-xl font-bold text-white">{{ skill.name }}</h2>
            <p class="text-primary-100 text-sm mt-1">{{ skill.id }}</p>
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
          <div class="space-y-2">
            <div v-for="dep in skill.dependencies" :key="dep.skill_id" class="flex items-center justify-between bg-gray-50 rounded-lg px-4 py-2">
              <span class="text-sm text-gray-700">{{ dep.skill_id }}</span>
              <span class="text-xs text-gray-500">{{ dep.version }}</span>
            </div>
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
      </div>
      
      <div class="border-t border-gray-100 px-6 py-4 flex items-center justify-between">
        <button class="btn btn-secondary" @click="$emit('close')">
          <ArrowLeft class="w-4 h-4" />
          返回
        </button>
        
        <button
          :class="[
            'btn btn-lg flex items-center gap-2',
            skill.installed ? 'btn-secondary' : 'btn-primary'
          ]"
          @click="handleAction"
        >
          <component :is="skill.installed ? DownloadCheck : Download" class="w-5 h-5" />
          {{ skill.installed ? '已安装' : '安装技能' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import {
  X, FileText, User, GitBranch, Tag,
  Settings, Server, Calendar, ArrowLeft, Download, CheckCircle
} from 'lucide-vue-next'

const props = defineProps({
  skill: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close', 'install', 'uninstall'])

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
    emit('uninstall', props.skill.id)
  } else {
    emit('install', props.skill.id)
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
