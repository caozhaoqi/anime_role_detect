<template>
  <header class="bg-white border-b border-gray-100 sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center gap-6">
          <div class="flex items-center gap-3 cursor-pointer" @click="$emit('navigate', 'skills')">
            <div class="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
              <Wand class="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 class="text-xl font-bold text-gray-900">ARD Skill Hub</h1>
              <p class="text-xs text-gray-500">技能仓库</p>
            </div>
          </div>
          
          <!-- 导航链接 -->
          <nav class="hidden md:flex items-center gap-4">
            <button
              :class="[
                'px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                currentPage === 'skills' ? 'text-primary-600 bg-primary-50' : 'text-gray-600 hover:bg-gray-50'
              ]"
              @click="$emit('navigate', 'skills')"
            >
              <span class="flex items-center gap-2">
                <Package class="w-4 h-4" />
                技能列表
              </span>
            </button>
            <button
              :class="[
                'px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                currentPage === 'help' ? 'text-primary-600 bg-primary-50' : 'text-gray-600 hover:bg-gray-50'
              ]"
              @click="$emit('navigate', 'help')"
            >
              <span class="flex items-center gap-2">
                <HelpCircle class="w-4 h-4" />
                使用指南
              </span>
            </button>
            <button
              v-if="isLoggedIn && isDeveloper"
              :class="[
                'px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                currentPage === 'dashboard' ? 'text-primary-600 bg-primary-50' : 'text-gray-600 hover:bg-gray-50'
              ]"
              @click="$emit('navigate', 'dashboard')"
            >
              <span class="flex items-center gap-2">
                <Settings class="w-4 h-4" />
                开发者后台
              </span>
            </button>
          </nav>
        </div>
        
        <div class="flex-1 max-w-xl mx-8">
          <div v-if="currentPage === 'skills'" class="relative">
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
          
          <div v-if="isLoggedIn" class="flex items-center gap-3">
            <button
              class="btn btn-primary btn-sm flex items-center gap-2"
              @click="$emit('registerSkill')"
            >
              <Plus class="w-4 h-4" />
              <span>发布技能</span>
            </button>
            
            <div class="relative" @click="showUserMenu = !showUserMenu">
              <button class="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors">
                <div class="w-8 h-8 bg-gradient-to-br from-primary-400 to-primary-600 rounded-full flex items-center justify-center text-white font-medium">
                  {{ username.charAt(0).toUpperCase() }}
                </div>
                <span class="hidden sm:inline">{{ username }}</span>
                <ChevronDown class="w-4 h-4" />
              </button>
              
              <div v-if="showUserMenu" class="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-lg border border-gray-100 py-2 z-50">
                <button
                  class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                  @click="$emit('viewProfile')"
                >
                  <User class="w-4 h-4" />
                  <span>个人资料</span>
                </button>
                <button
                  class="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                  @click="$emit('viewSkills')"
                >
                  <Package class="w-4 h-4" />
                  <span>我的技能</span>
                </button>
                <hr class="my-2 border-gray-100" />
                <button
                  class="w-full px-4 py-2 text-left text-sm text-red-500 hover:bg-gray-50 flex items-center gap-2"
                  @click="$emit('logout')"
                >
                  <LogOut class="w-4 h-4" />
                  <span>退出登录</span>
                </button>
              </div>
            </div>
          </div>
          
          <div v-else class="flex items-center gap-2">
            <button
              class="btn btn-outline btn-sm"
              @click="$emit('login')"
            >
              登录
            </button>
            <button
              class="btn btn-primary btn-sm"
              @click="$emit('register')"
            >
              注册
            </button>
          </div>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Wand, Search, Package, Tag, Plus, ChevronDown, User, LogOut, HelpCircle, Settings } from 'lucide-vue-next'

const props = defineProps({
  stats: {
    type: Object,
    default: () => ({})
  },
  currentPage: {
    type: String,
    default: 'skills'
  },
  isDeveloper: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['search', 'login', 'register', 'registerSkill', 'logout', 'viewProfile', 'viewSkills', 'navigate'])

const searchInput = ref('')
const showUserMenu = ref(false)
const isLoggedIn = ref(false)
const username = ref('')

const checkLoginStatus = () => {
  const token = localStorage.getItem('token')
  const user = localStorage.getItem('username')
  isLoggedIn.value = !!token
  username.value = user || ''
}

const handleSearch = () => {
  emit('search', searchInput.value)
}

onMounted(() => {
  checkLoginStatus()
})

defineExpose({
  checkLoginStatus
})
</script>

