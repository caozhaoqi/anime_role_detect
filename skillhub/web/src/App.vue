<template>
  <div class="min-h-screen bg-gray-50">
    <Header 
      ref="headerRef"
      :stats="stats" 
      :currentPage="currentPage"
      @search="handleSearch"
      @login="showLoginModal = true"
      @register="showRegisterModal = true"
      @registerSkill="showSkillRegisterModal = true"
      @logout="handleLogout"
      @navigate="currentPage = $event"
    />
    
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- 技能列表页面 -->
      <template v-if="currentPage === 'skills'">
        <div class="flex flex-col lg:flex-row gap-6">
          <aside class="lg:w-64 flex-shrink-0">
            <CategoryFilter 
              :categories="categories"
              :selectedCategory="selectedCategory"
              @select="selectedCategory = $event"
            />
          </aside>
          
          <div class="flex-1">
            <SkillList 
              :skills="skills"
              :loading="loading"
              :searchKeyword="searchKeyword"
              @view-detail="viewSkillDetail"
            />
          </div>
        </div>
      </template>
      
      <!-- 帮助页面 -->
      <HelpPage v-else-if="currentPage === 'help'" />
    </main>
    
    <SkillDetail 
      v-if="selectedSkill"
      :skill="selectedSkill"
      @close="selectedSkill = null"
      @install="handleInstallSkill"
      @uninstall="handleUninstallSkill"
      @update="handleUpdateSkill"
    />
    
    <RegisterSkill 
      v-if="showSkillRegisterModal"
      @close="showSkillRegisterModal = false"
      @success="handleSkillRegistered"
    />
    
    <!-- 用户登录模态框 -->
    <LoginModal 
      v-if="showLoginModal"
      @success="handleLoginSuccess"
      @switchToRegister="showLoginModal = false; showRegisterModal = true"
    />
    
    <!-- 用户注册模态框 -->
    <RegisterModal 
      v-if="showRegisterModal"
      @success="handleRegisterSuccess"
      @switchToLogin="showRegisterModal = false; showLoginModal = true"
    />
    
    <footer class="bg-white border-t border-gray-100 py-8 mt-12">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-500 text-sm">
        <p>ARD Skill Hub - Anime Role Detect Skill Repository</p>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import Header from './components/Header.vue'
import CategoryFilter from './components/CategoryFilter.vue'
import SkillList from './components/SkillList.vue'
import SkillDetail from './components/SkillDetail.vue'
import RegisterSkill from './components/RegisterSkill.vue'
import LoginModal from './components/LoginModal.vue'
import RegisterModal from './components/RegisterModal.vue'
import HelpPage from './components/HelpPage.vue'
import { skillApi } from './api/skillApi'
import { authApi } from './api/authApi'

const headerRef = ref(null)
const currentPage = ref('skills')
const stats = ref({})
const categories = ref([])
const skills = ref([])
const loading = ref(false)
const searchKeyword = ref('')
const selectedCategory = ref(null)
const selectedSkill = ref(null)
const showLoginModal = ref(false)
const showRegisterModal = ref(false)
const showSkillRegisterModal = ref(false)

const loadStats = async () => {
  try {
    const response = await skillApi.getStats()
    stats.value = response.data
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
}

const loadCategories = async () => {
  try {
    const response = await skillApi.getCategories()
    categories.value = Object.entries(response.data).map(([name, count]) => ({
      name,
      count,
      label: getCategoryLabel(name)
    }))
  } catch (error) {
    console.error('Failed to load categories:', error)
  }
}

const loadSkills = async () => {
  loading.value = true
  try {
    let response
    if (searchKeyword.value) {
      response = await skillApi.searchSkills({
        keyword: searchKeyword.value,
        category: selectedCategory.value || undefined
      })
      skills.value = response.data.skills
    } else {
      response = await skillApi.getSkills({
        category: selectedCategory.value || undefined
      })
      skills.value = response.data.skills
    }
  } catch (error) {
    console.error('Failed to load skills:', error)
    skills.value = []
  } finally {
    loading.value = false
  }
}

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

const handleSearch = (keyword) => {
  searchKeyword.value = keyword
}

const viewSkillDetail = async (skillId) => {
  try {
    const response = await skillApi.getSkill(skillId)
    selectedSkill.value = response.data
  } catch (error) {
    console.error('Failed to load skill detail:', error)
  }
}

const handleInstallSkill = async (skillId) => {
  try {
    // 显示加载状态 - 查找包含"安装技能"文本的按钮
    const buttons = document.querySelectorAll('.btn-primary')
    let installBtn = null
    buttons.forEach(btn => {
      if (btn.textContent.includes('安装技能')) {
        installBtn = btn
      }
    })
    if (installBtn) installBtn.disabled = true
    
    // 调用安装接口
    await skillApi.installSkill(skillId)
    
    // 触发文件下载
    const downloadUrl = `/api/skills/${skillId}/download`
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `${skillId}.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // 更新状态
    if (selectedSkill.value) {
      selectedSkill.value.installed = true
    }
    await loadSkills()
    
    // 显示成功提示
    alert(`技能 ${skillId} 安装成功！技能包已开始下载。`)
  } catch (error) {
    console.error('Failed to install skill:', error)
    alert(`安装失败: ${error.message || '未知错误'}`)
  } finally {
    // 恢复按钮状态
    const buttons = document.querySelectorAll('.btn-primary')
    buttons.forEach(btn => {
      if (btn.textContent.includes('安装技能')) {
        btn.disabled = false
      }
    })
  }
}

const handleUninstallSkill = async (skillId) => {
  try {
    await skillApi.uninstallSkill(skillId)
    if (selectedSkill.value) {
      selectedSkill.value.installed = false
    }
    await loadSkills()
  } catch (error) {
    console.error('Failed to uninstall skill:', error)
  }
}

const handleUpdateSkill = async (skillId) => {
  try {
    // 显示加载状态
    const buttons = document.querySelectorAll('.btn-primary')
    buttons.forEach(btn => {
      if (btn.textContent.includes('更新技能')) {
        btn.disabled = true
      }
    })
    
    // 调用安装接口（更新相当于重新安装）
    await skillApi.installSkill(skillId)
    
    // 触发文件下载
    const downloadUrl = `/api/skills/${skillId}/download`
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = `${skillId}.zip`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    // 更新状态
    await loadSkills()
    
    // 重新加载技能详情
    if (selectedSkill.value) {
      const response = await fetch(`/api/skills/${skillId}`)
      selectedSkill.value = await response.json()
    }
    
    // 显示成功提示
    alert(`技能 ${skillId} 更新成功！最新版本已开始下载。`)
  } catch (error) {
    console.error('Failed to update skill:', error)
    alert(`更新失败: ${error.message || '未知错误'}`)
  } finally {
    // 恢复按钮状态
    const buttons = document.querySelectorAll('.btn-primary')
    buttons.forEach(btn => {
      if (btn.textContent.includes('更新技能')) {
        btn.disabled = false
      }
    })
  }
}

const handleSkillRegistered = () => {
  showSkillRegisterModal.value = false
  loadSkills()
  loadCategories()
  loadStats()
}

const handleLoginSuccess = (user) => {
  showLoginModal.value = false
  if (headerRef.value) {
    headerRef.value.checkLoginStatus()
  }
  alert(`欢迎回来, ${user.username}!`)
}

const handleRegisterSuccess = (result) => {
  showRegisterModal.value = false
  showLoginModal.value = true
  alert('注册成功，请登录')
}

const handleLogout = async () => {
  try {
    await authApi.logout()
  } catch (error) {
    console.error('Logout failed:', error)
  } finally {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    if (headerRef.value) {
      headerRef.value.checkLoginStatus()
    }
    alert('已退出登录')
  }
}

watch([searchKeyword, selectedCategory], () => {
  loadSkills()
})

onMounted(() => {
  loadStats()
  loadCategories()
  loadSkills()
})
</script>
