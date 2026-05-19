<template>
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
    <div class="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
      <div class="bg-gradient-to-r from-primary-500 to-primary-600 px-6 py-8">
        <div class="text-center">
          <div class="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Wand class="w-8 h-8 text-white" />
          </div>
          <h2 class="text-2xl font-bold text-white">欢迎回来</h2>
          <p class="text-primary-100 mt-2">登录您的 ARD Skill Hub 账户</p>
        </div>
      </div>
      
      <div class="px-6 py-6">
        <form @submit.prevent="handleLogin" class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">用户名</label>
            <div class="relative">
              <User class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                v-model="form.username"
                type="text"
                placeholder="请输入用户名"
                class="w-full pl-10 pr-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                required
              />
            </div>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">密码</label>
            <div class="relative">
              <Lock class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                v-model="form.password"
                :type="showPassword ? 'text' : 'password'"
                placeholder="请输入密码"
                class="w-full pl-10 pr-12 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                required
              />
              <button
                type="button"
                class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                @click="showPassword = !showPassword"
              >
                <Eye v-if="!showPassword" class="w-4 h-4" />
                <EyeOff v-else class="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <button
            type="submit"
            :disabled="loading"
            class="w-full py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white font-medium rounded-xl hover:from-primary-600 hover:to-primary-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
            <span>{{ loading ? '登录中...' : '登录' }}</span>
          </button>
          
          <p v-if="error" class="text-red-500 text-sm text-center">{{ error }}</p>
        </form>
        
        <div class="mt-6 pt-4 border-t border-gray-100">
          <p class="text-center text-gray-500 text-sm">
            还没有账户？
            <button
              class="text-primary-500 hover:text-primary-600 font-medium"
              @click="$emit('switchToRegister')"
            >
              立即注册
            </button>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { Wand, User, Lock, Eye, EyeOff, Loader2 } from 'lucide-vue-next'
import { authApi } from '../api/authApi'

defineEmits(['success', 'switchToRegister'])

const emit = defineEmits(['success', 'switchToRegister'])

const form = reactive({
  username: '',
  password: ''
})

const showPassword = ref(false)
const loading = ref(false)
const error = ref('')

const handleLogin = async () => {
  loading.value = true
  error.value = ''
  
  try {
    const response = await authApi.login(form)
    if (response.data.success) {
      localStorage.setItem('token', response.data.token)
      localStorage.setItem('username', response.data.username)
      emit('success', response.data)
    } else {
      error.value = response.data.message
    }
  } catch (err) {
    error.value = '登录失败，请稍后重试'
  } finally {
    loading.value = false
  }
}
</script>
