<template>
  <div class="min-h-screen bg-gray-50">
    <Header 
      :stats="stats" 
      @search="handleSearch"
      @register="showRegisterModal = true"
    />
    
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
    </main>
    
    <SkillDetail 
      v-if="selectedSkill"
      :skill="selectedSkill"
      @close="selectedSkill = null"
      @install="handleInstallSkill"
      @uninstall="handleUninstallSkill"
    />
    
    <RegisterSkill 
      v-if="showRegisterModal"
      @close="showRegisterModal = false"
      @success="handleSkillRegistered"
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
import { skillApi } from './api/skillApi'

const stats = ref({})
const categories = ref([])
const skills = ref([])
const loading = ref(false)
const searchKeyword = ref('')
const selectedCategory = ref(null)
const selectedSkill = ref(null)
const showRegisterModal = ref(false)

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
    await skillApi.installSkill(skillId)
    if (selectedSkill.value) {
      selectedSkill.value.installed = true
    }
    await loadSkills()
  } catch (error) {
    console.error('Failed to install skill:', error)
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

const handleSkillRegistered = () => {
  showRegisterModal.value = false
  loadSkills()
  loadCategories()
  loadStats()
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
