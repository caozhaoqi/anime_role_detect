<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <div>
        <h2 class="text-lg font-semibold text-gray-900">技能列表</h2>
        <p class="text-sm text-gray-500 mt-1">
          {{ skills.length }} 个技能
          <span v-if="searchKeyword" class="text-primary-600">
            - 搜索: "{{ searchKeyword }}"
          </span>
        </p>
      </div>
      
      <div class="flex items-center gap-2">
        <button
          :class="[
            'btn btn-sm',
            viewMode === 'grid' ? 'btn-primary' : 'btn-secondary'
          ]"
          @click="viewMode = 'grid'"
        >
          <LayoutGrid class="w-4 h-4" />
        </button>
        <button
          :class="[
            'btn btn-sm',
            viewMode === 'list' ? 'btn-primary' : 'btn-secondary'
          ]"
          @click="viewMode = 'list'"
        >
          <List class="w-4 h-4" />
        </button>
      </div>
    </div>
    
    <div v-if="loading" class="flex justify-center py-16">
      <div class="w-10 h-10 border-4 border-primary-200 border-t-primary-500 rounded-full animate-spin"></div>
    </div>
    
    <div v-else-if="skills.length === 0" class="text-center py-16">
      <Package class="w-16 h-16 text-gray-300 mx-auto mb-4" />
      <h3 class="text-lg font-medium text-gray-900 mb-2">未找到技能</h3>
      <p class="text-gray-500">尝试更换搜索关键词或筛选条件</p>
    </div>
    
    <div v-else :class="['grid gap-4', viewMode === 'grid' ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' : 'grid-cols-1']">
      <div
        v-for="skill in skills"
        :key="skill.name"
        :class="[
          'skill-card animate-fade-in cursor-pointer',
          viewMode === 'list' ? 'flex items-start gap-4' : ''
        ]"
        @click="$emit('view-detail', skill.name)"
      >
        <div :class="['flex-1', viewMode === 'grid' ? '' : '']">
          <div class="flex items-start justify-between gap-3 mb-3">
            <div>
              <h3 class="font-semibold text-gray-900 text-base">{{ skill.name }}</h3>
              <p class="text-sm text-gray-500 mt-0.5">{{ skill.id }}</p>
            </div>
            <div class="flex items-center gap-2">
              <span :class="['status-tag', `status-${skill.status}`]">
                {{ getStatusLabel(skill.status) }}
              </span>
              <button
                v-if="isLoggedIn"
                class="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                @click.stop="toggleFavorite(skill.name)"
              >
                <Heart 
                  :class="[
                    'w-4 h-4',
                    favorites[skill.name] ? 'text-red-500 fill-red-500' : 'text-gray-400'
                  ]"
                />
              </button>
            </div>
          </div>
          
          <p class="text-sm text-gray-600 mb-3 line-clamp-2">{{ skill.description }}</p>
          
          <div class="flex flex-wrap items-center gap-2 mb-3">
            <span :class="['category-badge', `category-${skill.category}`]">
              {{ getCategoryLabel(skill.category) }}
            </span>
            <span v-for="tag in skill.tags.slice(0, 3)" :key="tag" class="tag tag-category">
              {{ tag }}
            </span>
            <span v-if="skill.tags.length > 3" class="tag tag-category text-gray-400">
              +{{ skill.tags.length - 3 }}
            </span>
          </div>
          
          <div class="flex items-center justify-between text-xs text-gray-500 mb-3">
            <div class="flex items-center gap-3">
              <span class="flex items-center gap-1">
                <Download class="w-3.5 h-3.5" />
                {{ skill.downloads || 0 }}
              </span>
              <span class="flex items-center gap-1">
                <Star class="w-3.5 h-3.5 text-yellow-400" />
                {{ skill.rating || '0' }}
              </span>
            </div>
          </div>
        </div>
        
        <div v-if="viewMode === 'list'" class="flex-shrink-0">
          <button
            :class="[
              'btn btn-sm',
              skill.installed ? 'btn-secondary' : 'btn-primary'
            ]"
            @click.stop="$emit('view-detail', skill.id)"
          >
            {{ skill.installed ? '已安装' : '安装' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { LayoutGrid, List, Package, User, Tag, Heart, Download, Star } from 'lucide-vue-next'

defineProps({
  skills: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  searchKeyword: {
    type: String,
    default: ''
  }
})

defineEmits(['view-detail'])

const viewMode = ref('grid')
const isLoggedIn = ref(false)
const favorites = ref({})

const loadFavorites = async () => {
  try {
    const token = localStorage.getItem('token')
    if (!token) {
      isLoggedIn.value = false
      return
    }
    isLoggedIn.value = true
    
    const response = await fetch('/api/favorites', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    const data = await response.json()
    data.skills.forEach(skill => {
      favorites.value[skill.name] = true
    })
  } catch (error) {
    console.error('Failed to load favorites:', error)
  }
}

const toggleFavorite = async (skillName) => {
  try {
    const token = localStorage.getItem('token')
    const isFav = favorites.value[skillName]
    
    const method = isFav ? 'DELETE' : 'POST'
    const response = await fetch(`/api/skills/${skillName}/favorite`, {
      method,
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    
    const result = await response.json()
    if (result.success) {
      favorites.value[skillName] = !isFav
    } else {
      alert(result.message)
    }
  } catch (error) {
    console.error('Failed to toggle favorite:', error)
    alert('操作失败')
  }
}

onMounted(() => {
  loadFavorites()
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
    stable: '稳定',
    testing: '测试',
    development: '开发',
    deprecated: '弃用'
  }
  return labels[status] || status
}
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.status-tag {
  @apply px-2 py-1 rounded-full text-xs font-medium;
}

.status-stable { @apply bg-green-100 text-green-700; }
.status-testing { @apply bg-yellow-100 text-yellow-700; }
.status-development { @apply bg-blue-100 text-blue-700; }
.status-deprecated { @apply bg-red-100 text-red-700; }
</style>
