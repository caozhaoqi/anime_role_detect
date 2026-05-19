<template>
  <div class="bg-white rounded-xl shadow-sm border border-gray-100 p-5">
    <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
      <LayoutGrid class="w-5 h-5 text-primary-500" />
      技能分类
    </h3>
    
    <nav class="space-y-1">
      <button
        :class="[
          'w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-all',
          selectedCategory === null
            ? 'bg-primary-50 text-primary-700'
            : 'text-gray-600 hover:bg-gray-50'
        ]"
        @click="$emit('select', null)"
      >
        <div class="flex items-center justify-between">
          <span>全部技能</span>
        </div>
      </button>
      
      <button
        v-for="category in categories"
        :key="category.name"
        :class="[
          'w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-all',
          selectedCategory === category.name
            ? 'bg-primary-50 text-primary-700'
            : 'text-gray-600 hover:bg-gray-50'
        ]"
        @click="$emit('select', category.name)"
      >
        <div class="flex items-center justify-between">
          <span>{{ category.label }}</span>
          <span class="text-gray-400 text-xs">{{ category.count }}</span>
        </div>
      </button>
    </nav>
    
    <div class="mt-6 pt-5 border-t border-gray-100">
      <h3 class="font-semibold text-gray-900 mb-3 flex items-center gap-2">
        <Star class="w-5 h-5 text-primary-500" />
        快捷筛选
      </h3>
      
      <div class="space-y-2">
        <button
          v-for="status in statusFilters"
          :key="status.value"
          :class="[
            'w-full text-left px-3 py-2 rounded-lg text-sm transition-all',
            'text-gray-600 hover:bg-gray-50'
          ]"
        >
          <div class="flex items-center gap-2">
            <span :class="['status-dot', `status-dot-${status.value}`]"></span>
            <span>{{ status.label }}</span>
          </div>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { LayoutGrid, Star } from 'lucide-vue-next'

defineProps({
  categories: {
    type: Array,
    default: () => []
  },
  selectedCategory: {
    type: String,
    default: null
  }
})

defineEmits(['select'])

const statusFilters = [
  { label: '稳定版', value: 'stable' },
  { label: '测试中', value: 'testing' },
  { label: '开发中', value: 'development' },
  { label: '已弃用', value: 'deprecated' }
]
</script>
