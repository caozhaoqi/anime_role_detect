<template>
  <header class="bg-white border-b border-gray-100 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
            <Wand class="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 class="text-xl font-bold text-gray-900">ARD Skill Hub</h1>
            <p class="text-xs text-gray-500">技能仓库</p>
          </div>
        </div>
        
        <div class="flex-1 max-w-xl mx-8">
          <div class="relative">
            <Search class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              v-model="searchInput"
              type="text"
              placeholder="搜索技能..."
              class="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
              @keyup.enter="handleSearch"
            />
          </div>
        </div>
        
        <div class="flex items-center gap-4">
          <div class="hidden md:flex items-center gap-6 text-sm text-gray-600">
            <div class="flex items-center gap-2">
              <Package class="w-4 h-4 text-primary-500" />
              <span>{{ stats.total_skills || 0 }} 技能</span>
            </div>
            <div class="flex items-center gap-2">
              <Tag class="w-4 h-4 text-primary-500" />
              <span>{{ stats.total_categories || 0 }} 分类</span>
            </div>
          </div>
          
          <button
            class="btn btn-primary btn-sm flex items-center gap-2"
            @click="$emit('register')"
          >
            <Plus class="w-4 h-4" />
            <span>发布技能</span>
          </button>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref } from 'vue'
import { Wand, Search, Package, Tag, Plus } from 'lucide-vue-next'

defineProps({
  stats: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['search', 'register'])

const searchInput = ref('')

const handleSearch = () => {
  emit('search', searchInput.value)
}
</script>
