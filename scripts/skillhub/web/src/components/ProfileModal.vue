<template>
  <Teleport to="body">
    <div v-if="show" class="fixed inset-0 z-50 flex items-center justify-center">
      <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="$emit('close')"></div>
      
      <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col animate-slide-up">
        <!-- 头部 -->
        <div class="bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 px-6 py-5">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <div class="w-16 h-16 bg-white rounded-full flex items-center justify-center text-primary-600 text-2xl font-bold shadow-lg">
                {{ userInfo?.username?.charAt(0)?.toUpperCase() || 'U' }}
              </div>
              <div class="text-white">
                <h2 class="text-xl font-bold">{{ userInfo?.username || '用户' }}</h2>
                <p class="text-primary-100 text-sm">{{ getRoleLabel(userInfo?.role) }}</p>
              </div>
            </div>
            <button 
              @click="$emit('close')"
              class="w-10 h-10 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center text-white transition-colors"
            >
              <X class="w-5 h-5" />
            </button>
          </div>
          
          <!-- 统计卡片 -->
          <div class="grid grid-cols-4 gap-4 mt-6">
            <div class="bg-white/10 rounded-xl px-4 py-3 text-center">
              <div class="text-2xl font-bold text-white">{{ stats.installed }}</div>
              <div class="text-primary-100 text-xs">已安装</div>
            </div>
            <div class="bg-white/10 rounded-xl px-4 py-3 text-center">
              <div class="text-2xl font-bold text-white">{{ stats.favorites }}</div>
              <div class="text-primary-100 text-xs">我的收藏</div>
            </div>
            <div class="bg-white/10 rounded-xl px-4 py-3 text-center">
              <div class="text-2xl font-bold text-white">{{ stats.developed }}</div>
              <div class="text-primary-100 text-xs">发布技能</div>
            </div>
            <div class="bg-white/10 rounded-xl px-4 py-3 text-center">
              <div class="text-2xl font-bold text-white">{{ stats.updates }}</div>
              <div class="text-primary-100 text-xs">可更新</div>
            </div>
          </div>
        </div>
        
        <!-- 标签页 -->
        <div class="border-b border-gray-100 px-6 bg-gray-50">
          <div class="flex gap-1">
            <button
              v-for="tab in tabs"
              :key="tab.id"
              :class="[
                'px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                activeTab === tab.id 
                  ? 'text-primary-600 border-primary-600' 
                  : 'text-gray-500 border-transparent hover:text-gray-700'
              ]"
              @click="activeTab = tab.id"
            >
              <span class="flex items-center gap-2">
                <component :is="tab.icon" class="w-4 h-4" />
                {{ tab.label }}
              </span>
            </button>
          </div>
        </div>
        
        <!-- 内容区域 -->
        <div class="flex-1 overflow-y-auto p-6">
          <!-- 个人资料 -->
          <div v-if="activeTab === 'profile'" class="space-y-6">
            <div class="bg-gray-50 rounded-xl p-5">
              <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <User class="w-5 h-5 text-primary-500" />
                基本信息
              </h3>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="text-xs text-gray-500 uppercase tracking-wide">用户名</label>
                  <p class="text-gray-900 font-medium mt-1">{{ userInfo?.username || '未知' }}</p>
                </div>
                <div>
                  <label class="text-xs text-gray-500 uppercase tracking-wide">角色</label>
                  <p class="text-gray-900 font-medium mt-1">
                    <span :class="[
                      'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                      userInfo?.role === 'admin' ? 'bg-purple-100 text-purple-800' :
                      userInfo?.role === 'developer' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                    ]">
                      {{ getRoleLabel(userInfo?.role) }}
                    </span>
                  </p>
                </div>
                <div>
                  <label class="text-xs text-gray-500 uppercase tracking-wide">邮箱</label>
                  <p class="text-gray-900 font-medium mt-1">{{ userInfo?.email || '未设置' }}</p>
                </div>
                <div>
                  <label class="text-xs text-gray-500 uppercase tracking-wide">注册时间</label>
                  <p class="text-gray-900 font-medium mt-1">{{ userInfo?.created_at || '未知' }}</p>
                </div>
              </div>
            </div>
            
            <div class="bg-gray-50 rounded-xl p-5">
              <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Settings class="w-5 h-5 text-primary-500" />
                账号设置
              </h3>
              <div class="space-y-3">
                <button class="w-full flex items-center justify-between px-4 py-3 bg-white rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors">
                  <div class="flex items-center gap-3">
                    <Key class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-700">修改密码</span>
                  </div>
                  <ChevronRight class="w-5 h-5 text-gray-400" />
                </button>
                <button class="w-full flex items-center justify-between px-4 py-3 bg-white rounded-lg border border-gray-200 hover:border-primary-300 hover:bg-primary-50 transition-colors">
                  <div class="flex items-center gap-3">
                    <Bell class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-700">通知设置</span>
                  </div>
                  <ChevronRight class="w-5 h-5 text-gray-400" />
                </button>
                <button class="w-full flex items-center justify-between px-4 py-3 bg-white rounded-lg border border-gray-200 hover:border-red-300 hover:bg-red-50 transition-colors">
                  <div class="flex items-center gap-3">
                    <Shield class="w-5 h-5 text-gray-400" />
                    <span class="text-gray-700">隐私设置</span>
                  </div>
                  <ChevronRight class="w-5 h-5 text-gray-400" />
                </button>
              </div>
            </div>
          </div>
          
          <!-- 已安装技能 -->
          <div v-else-if="activeTab === 'installed'" class="space-y-4">
            <div v-if="installedSkills.length === 0" class="text-center py-12">
              <Package class="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p class="text-gray-500">暂无已安装的技能</p>
            </div>
            <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                v-for="skill in installedSkills" 
                :key="skill.name"
                class="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start justify-between mb-3">
                  <div>
                    <h4 class="font-semibold text-gray-900">{{ skill.name }}</h4>
                    <p class="text-sm text-gray-500">{{ skill.id }}</p>
                  </div>
                  <span class="status-tag" :class="`status-${skill.status}`">
                    {{ skill.status }}
                  </span>
                </div>
                <p class="text-sm text-gray-600 mb-3 line-clamp-2">{{ skill.description }}</p>
                <div class="flex items-center justify-between">
                  <span class="text-xs text-gray-400">v{{ skill.version }}</span>
                  <div class="flex gap-2">
                    <button class="text-xs text-primary-600 hover:text-primary-700">检查更新</button>
                    <button class="text-xs text-red-500 hover:text-red-600">卸载</button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 我的收藏 -->
          <div v-else-if="activeTab === 'favorites'" class="space-y-4">
            <div v-if="favoriteSkills.length === 0" class="text-center py-12">
              <Heart class="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p class="text-gray-500">暂无收藏的技能</p>
            </div>
            <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                v-for="skill in favoriteSkills" 
                :key="skill.name"
                class="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start justify-between mb-3">
                  <div>
                    <h4 class="font-semibold text-gray-900">{{ skill.name }}</h4>
                    <p class="text-sm text-gray-500">{{ skill.id }}</p>
                  </div>
                  <button class="text-red-500 hover:text-red-600">
                    <Heart class="w-5 h-5 fill-red-500" />
                  </button>
                </div>
                <p class="text-sm text-gray-600 mb-3 line-clamp-2">{{ skill.description }}</p>
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-1">
                    <Star class="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    <span class="text-sm text-gray-600">{{ skill.rating || 0 }}</span>
                  </div>
                  <button class="btn btn-primary btn-sm">安装</button>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 我的发布 -->
          <div v-else-if="activeTab === 'developed'" class="space-y-4">
            <div class="flex items-center justify-between mb-4">
              <h3 class="font-semibold text-gray-900">我发布的技能</h3>
              <button class="btn btn-primary btn-sm" @click="$emit('registerSkill')">
                <Plus class="w-4 h-4" />
                发布新技能
              </button>
            </div>
            <div v-if="developedSkills.length === 0" class="text-center py-12">
              <Wand class="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p class="text-gray-500">您还没有发布任何技能</p>
            </div>
            <div v-else class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div 
                v-for="skill in developedSkills" 
                :key="skill.name"
                class="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start justify-between mb-3">
                  <div>
                    <h4 class="font-semibold text-gray-900">{{ skill.name }}</h4>
                    <p class="text-sm text-gray-500">{{ skill.id }}</p>
                  </div>
                  <span class="status-tag" :class="`status-${skill.status}`">
                    {{ skill.status }}
                  </span>
                </div>
                <p class="text-sm text-gray-600 mb-3 line-clamp-2">{{ skill.description }}</p>
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-3 text-xs text-gray-500">
                    <span class="flex items-center gap-1">
                      <Download class="w-3 h-3" />
                      {{ skill.downloads || 0 }}
                    </span>
                    <span class="flex items-center gap-1">
                      <Star class="w-3 h-3" />
                      {{ skill.rating || 0 }}
                    </span>
                  </div>
                  <button class="btn btn-outline btn-sm">管理</button>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 更新通知 -->
          <div v-else-if="activeTab === 'updates'" class="space-y-4">
            <div v-if="updateNotifications.length === 0" class="text-center py-12">
              <CheckCircle class="w-16 h-16 text-green-300 mx-auto mb-4" />
              <p class="text-gray-500">所有技能都是最新版本</p>
            </div>
            <div v-else class="space-y-3">
              <div 
                v-for="(notification, index) in updateNotifications" 
                :key="index"
                class="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow"
              >
                <div class="flex items-start gap-4">
                  <div class="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <RefreshCw class="w-5 h-5 text-primary-600" />
                  </div>
                  <div class="flex-1">
                    <h4 class="font-semibold text-gray-900">{{ notification.skill_name }}</h4>
                    <p class="text-sm text-gray-500 mt-1">
                      {{ notification.current_version }} → {{ notification.latest_version }}
                    </p>
                    <p class="text-sm text-gray-600 mt-2">{{ notification.changelog }}</p>
                  </div>
                  <button class="btn btn-primary btn-sm self-center">
                    更新
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref, watch, onMounted } from 'vue'
import { 
  X, User, Settings, Package, Heart, Wand, Download, Star, Bell, Shield, Key, 
  ChevronRight, CheckCircle, RefreshCw, Plus
} from 'lucide-vue-next'

const props = defineProps({
  show: {
    type: Boolean,
    default: false
  },
  userInfo: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['close', 'registerSkill'])

const activeTab = ref('profile')
const tabs = [
  { id: 'profile', label: '个人资料', icon: User },
  { id: 'installed', label: '已安装', icon: Package },
  { id: 'favorites', label: '我的收藏', icon: Heart },
  { id: 'developed', label: '我的发布', icon: Wand },
  { id: 'updates', label: '更新通知', icon: RefreshCw }
]

const stats = ref({
  installed: 0,
  favorites: 0,
  developed: 0,
  updates: 0
})

const installedSkills = ref([])
const favoriteSkills = ref([])
const developedSkills = ref([])
const updateNotifications = ref([])

const getRoleLabel = (role) => {
  const labels = {
    admin: '管理员',
    developer: '开发者',
    user: '普通用户'
  }
  return labels[role] || '普通用户'
}

const loadUserData = async () => {
  try {
    const token = localStorage.getItem('token')
    if (!token) return
    
    // 加载收藏列表
    const favResponse = await fetch('/api/favorites', {
      headers: { 'Authorization': `Bearer ${token}` }
    })
    const favData = await favResponse.json()
    favoriteSkills.value = favData.skills || []
    stats.value.favorites = favoriteSkills.value.length
    
    // 加载已安装技能
    const installedResponse = await fetch('/api/skills?installed=true', {
      headers: { 'Authorization': `Bearer ${token}` }
    })
    const installedData = await installedResponse.json()
    installedSkills.value = installedData.skills || []
    stats.value.installed = installedSkills.value.length
    
    // 加载更新通知
    const updateResponse = await fetch('/api/notifications/check-updates', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    })
    const updateData = await updateResponse.json()
    updateNotifications.value = updateData.updates || []
    stats.value.updates = updateNotifications.value.length
    
    // 开发者技能（这里简化处理）
    developedSkills.value = []
    stats.value.developed = 0
    
  } catch (error) {
    console.error('Failed to load user data:', error)
  }
}

watch(() => props.show, (newVal) => {
  if (newVal) {
    loadUserData()
  }
})

onMounted(() => {
  if (props.show) {
    loadUserData()
  }
})
</script>

<style scoped>
.animate-slide-up {
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>