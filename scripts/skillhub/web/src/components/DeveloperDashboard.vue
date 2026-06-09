<template>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- 页面标题 -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 flex items-center gap-3">
        <Settings class="w-7 h-7 text-primary-500" />
        开发者后台
      </h1>
      <p class="text-gray-500 mt-2">管理您的技能和版本发布</p>
    </div>

    <!-- 技能列表 -->
    <div class="bg-white rounded-2xl shadow-lg p-6 mb-8">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-xl font-bold text-gray-900 flex items-center gap-2">
          <Package class="w-5 h-5 text-primary-500" />
          我的技能
        </h2>
        <button
          class="btn btn-primary flex items-center gap-2"
          @click="showCreateSkill = true"
        >
          <Plus class="w-4 h-4" />
          发布新技能
        </button>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div
          v-for="skill in skills"
          :key="skill.id"
          class="border border-gray-100 rounded-xl p-4 hover:border-primary-200 transition-colors cursor-pointer"
          @click="selectSkill(skill)"
        >
          <div class="flex items-start justify-between">
            <div>
              <h3 class="font-medium text-gray-900">{{ skill.name }}</h3>
              <p class="text-sm text-gray-500">{{ skill.description }}</p>
            </div>
            <span class="px-2 py-1 bg-primary-100 text-primary-700 text-xs font-medium rounded-full">
              v{{ skill.version }}
            </span>
          </div>
          <div class="flex items-center gap-4 mt-3 text-sm text-gray-400">
            <span class="flex items-center gap-1">
              <Download class="w-4 h-4" />
              {{ skill.downloads }}
            </span>
            <span class="flex items-center gap-1">
              <Calendar class="w-4 h-4" />
              {{ skill.updated_at }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- 技能详情与版本管理 -->
    <div v-if="selectedSkill" class="bg-white rounded-2xl shadow-lg p-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h2 class="text-xl font-bold text-gray-900">{{ selectedSkill.name }}</h2>
          <p class="text-gray-500 mt-1">{{ selectedSkill.description }}</p>
        </div>
        <button
          class="btn btn-secondary"
          @click="selectedSkill = null"
        >
          返回列表
        </button>
      </div>

      <!-- 版本列表 -->
      <div class="mb-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="font-medium text-gray-900 flex items-center gap-2">
            <Clock class="w-4 h-4 text-gray-500" />
            版本历史
          </h3>
          <div class="flex items-center gap-2">
            <button
              class="btn btn-secondary btn-sm flex items-center gap-2"
              @click="showUploadPackage = true"
            >
              <Upload class="w-4 h-4" />
              上传更新包
            </button>
            <button
              class="btn btn-primary btn-sm flex items-center gap-2"
              @click="showCreateVersion = true"
            >
              <Plus class="w-4 h-4" />
              发布新版本
            </button>
          </div>
        </div>

        <div class="space-y-3">
          <div
            v-for="(version, index) in skillVersions"
            :key="version.version"
            :class="[
              'flex items-start gap-4 p-4 rounded-lg',
              index === 0 ? 'bg-primary-50 border border-primary-100' : 'bg-gray-50'
            ]"
          >
            <div class="flex-shrink-0">
              <span :class="[
                'px-3 py-1.5 rounded-full text-sm font-medium',
                index === 0 ? 'bg-primary-100 text-primary-700' : 'bg-gray-200 text-gray-700'
              ]">
                v{{ version.version }}
              </span>
              <span v-if="index === 0" class="ml-2 text-xs text-primary-600 font-medium">最新</span>
            </div>
            <div class="flex-1">
              <p class="text-sm text-gray-600">{{ version.changelog }}</p>
              <p class="text-xs text-gray-400 mt-1">发布于 {{ version.release_date }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- 技能信息 -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-gray-50 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-primary-600">{{ selectedSkill.downloads }}</div>
          <div class="text-sm text-gray-500 mt-1">总下载量</div>
        </div>
        <div class="bg-gray-50 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-primary-600">{{ skillVersions.length }}</div>
          <div class="text-sm text-gray-500 mt-1">版本数量</div>
        </div>
        <div class="bg-gray-50 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-primary-600">{{ selectedSkill.version }}</div>
          <div class="text-sm text-gray-500 mt-1">当前版本</div>
        </div>
        <div class="bg-gray-50 rounded-xl p-4 text-center">
          <div class="text-2xl font-bold text-primary-600">{{ getStatusLabel(selectedSkill.status) }}</div>
          <div class="text-sm text-gray-500 mt-1">状态</div>
        </div>
      </div>
    </div>

    <!-- 创建技能弹窗 -->
    <div v-if="showCreateSkill" class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="showCreateSkill = false"></div>
      <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-lg p-6 max-h-[90vh] overflow-y-auto">
        <h3 class="text-xl font-bold text-gray-900 mb-4">发布新技能</h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">技能名称</label>
            <input
              v-model="newSkill.name"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="如: ardc-my-skill"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">版本号</label>
            <input
              v-model="newSkill.version"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="如: 1.0.0"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">描述</label>
            <textarea
              v-model="newSkill.description"
              rows="3"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="技能描述"
            ></textarea>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">分类</label>
            <select
              v-model="newSkill.category"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="collector">数据采集</option>
              <option value="cleaner">数据清洗</option>
              <option value="trainer">模型训练</option>
              <option value="classifier">分类识别</option>
              <option value="search">搜索检索</option>
            </select>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">标签（逗号分隔）</label>
            <input
              v-model="newSkill.tags"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="tag1, tag2, tag3"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">依赖（逗号分隔）</label>
            <input
              v-model="newSkill.dependencies"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="package1, package2"
            />
          </div>
        </div>

        <div class="flex items-center justify-end gap-3 mt-6">
          <button
            class="btn btn-secondary"
            @click="showCreateSkill = false"
          >
            取消
          </button>
          <button
            class="btn btn-primary"
            @click="createSkill"
            :disabled="!newSkill.name || !newSkill.version"
          >
            发布技能
          </button>
        </div>
      </div>
    </div>

    <!-- 创建版本弹窗 -->
    <div v-if="showCreateVersion" class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="showCreateVersion = false"></div>
      <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-md p-6">
        <h3 class="text-xl font-bold text-gray-900 mb-4">发布新版本</h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">版本号</label>
            <input
              v-model="newVersion.version"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="如: 1.1.0"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">更新日志</label>
            <textarea
              v-model="newVersion.changelog"
              rows="3"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="描述本次更新的内容..."
            ></textarea>
          </div>
        </div>

        <div class="flex items-center justify-end gap-3 mt-6">
          <button
            class="btn btn-secondary"
            @click="showCreateVersion = false"
          >
            取消
          </button>
          <button
            class="btn btn-primary"
            @click="createVersion"
            :disabled="!newVersion.version"
          >
            发布版本
          </button>
        </div>
      </div>
    </div>

    <!-- 上传更新包弹窗 -->
    <div v-if="showUploadPackage" class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="showUploadPackage = false"></div>
      <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-md p-6">
        <h3 class="text-xl font-bold text-gray-900 mb-4">上传技能更新包</h3>

        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">技能名称</label>
            <input
              v-model="uploadData.skillName"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-gray-50"
              :placeholder="selectedSkill?.name"
              readonly
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">版本号</label>
            <input
              v-model="uploadData.version"
              type="text"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="如: 1.1.0"
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">选择更新包文件</label>
            <div
              class="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-primary-400 transition-colors cursor-pointer"
              @click="triggerFileUpload"
              @dragover.prevent="isDragging = true"
              @dragleave="isDragging = false"
              @drop.prevent="handleFileDrop"
              :class="{ 'border-primary-400 bg-primary-50': isDragging }"
            >
              <input
                ref="fileInputRef"
                type="file"
                class="hidden"
                accept=".zip,.tar,.gz,.tar.gz"
                @change="handleFileSelect"
              />
              <Upload class="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p class="text-gray-600 mb-1">
                点击或拖拽文件到此处上传
              </p>
              <p class="text-sm text-gray-400">
                支持 .zip, .tar.gz 格式
              </p>
              <p v-if="uploadData.file" class="mt-3 text-sm text-primary-600 font-medium">
                已选择: {{ uploadData.file.name }}
              </p>
            </div>
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">更新说明</label>
            <textarea
              v-model="uploadData.changelog"
              rows="3"
              class="w-full px-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="描述本次更新的内容..."
            ></textarea>
          </div>
        </div>

        <div v-if="uploadProgress > 0" class="mt-4">
          <div class="flex items-center justify-between text-sm text-gray-600 mb-1">
            <span>上传进度</span>
            <span>{{ uploadProgress }}%</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
              class="bg-primary-500 h-2 rounded-full transition-all duration-300"
              :style="{ width: uploadProgress + '%' }"
            ></div>
          </div>
        </div>

        <div class="flex items-center justify-end gap-3 mt-6">
          <button
            class="btn btn-secondary"
            @click="showUploadPackage = false"
          >
            取消
          </button>
          <button
            class="btn btn-primary"
            @click="doUploadPackage"
            :disabled="!uploadData.file || !uploadData.version || uploading"
          >
            {{ uploading ? '上传中...' : '上传并发布' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
  Settings, Package, Plus, Download, Calendar, Clock, Upload
} from 'lucide-vue-next'

const skills = ref([])
const selectedSkill = ref(null)
const skillVersions = ref([])
const showCreateSkill = ref(false)
const showCreateVersion = ref(false)
const showUploadPackage = ref(false)
const isDragging = ref(false)
const uploading = ref(false)
const uploadProgress = ref(0)
const fileInputRef = ref(null)

const newSkill = ref({
  name: '',
  version: '',
  description: '',
  category: 'collector',
  tags: '',
  dependencies: ''
})

const newVersion = ref({
  version: '',
  changelog: ''
})

const uploadData = ref({
  skillName: '',
  version: '',
  changelog: '',
  file: null
})

const loadSkills = async () => {
  try {
    console.log('=== Loading skills ===')
    const response = await fetch('/api/skills')
    const data = await response.json()
    skills.value = data.skills || []
    console.log('Skills loaded:', skills.value.length, 'skills')
    console.log('First skill:', skills.value[0])
  } catch (error) {
    console.error('Failed to load skills:', error)
  }
}

const selectSkill = async (skill) => {
  console.log('=== selectSkill called ===')
  console.log('Skill object:', skill)
  console.log('Skill id:', skill.id)
  console.log('Skill name:', skill.name)
  
  // 使用新对象触发响应式更新
  selectedSkill.value = null
  await new Promise(resolve => setTimeout(resolve, 0))
  selectedSkill.value = { ...skill }
  uploadData.value.skillName = skill.id
  
  await loadVersions(skill.id)
  
  console.log('=== After selection ===')
  console.log('selectedSkill.value:', selectedSkill.value)
  console.log('selectedSkill.value is truthy:', !!selectedSkill.value)
}

const loadVersions = async (skillName) => {
  try {
    const response = await fetch(`/api/skills/${skillName}/versions`)
    skillVersions.value = await response.json()
  } catch (error) {
    console.error('Failed to load versions:', error)
  }
}

const createSkill = async () => {
  try {
    const skillData = {
      name: newSkill.value.name,
      version: newSkill.value.version,
      description: newSkill.value.description,
      author: 'Developer',
      category: newSkill.value.category,
      tags: newSkill.value.tags.split(',').map(t => t.trim()).filter(Boolean),
      dependencies: newSkill.value.dependencies.split(',').map(d => d.trim()).filter(Boolean)
    }

    const token = localStorage.getItem('token')
    const response = await fetch('/api/skills', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(skillData)
    })

    const result = await response.json()
    if (result.success) {
      alert('技能发布成功！')
      showCreateSkill.value = false
      newSkill.value = {
        name: '',
        version: '',
        description: '',
        category: 'collector',
        tags: '',
        dependencies: ''
      }
      await loadSkills()
    } else {
      alert(result.message || '发布失败')
    }
  } catch (error) {
    console.error('Failed to create skill:', error)
    alert('发布失败: ' + error.message)
  }
}

const createVersion = async () => {
  try {
    const token = localStorage.getItem('token')
    const response = await fetch(
      `/api/skills/${selectedSkill.value.id}/versions?version=${newVersion.value.version}&changelog=${encodeURIComponent(newVersion.value.changelog)}`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    )

    const result = await response.json()
    if (result.success) {
      alert(`版本 ${newVersion.value.version} 发布成功！`)
      showCreateVersion.value = false
      newVersion.value = {
        version: '',
        changelog: ''
      }
      await loadVersions(selectedSkill.value.id)
      await loadSkills()
    } else {
      alert(result.message || '发布失败')
    }
  } catch (error) {
    console.error('Failed to create version:', error)
    alert('发布失败: ' + error.message)
  }
}

const triggerFileUpload = () => {
  fileInputRef.value?.click()
}

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    uploadData.value.file = file
  }
}

const handleFileDrop = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && (file.name.endsWith('.zip') || file.name.endsWith('.tar.gz') || file.name.endsWith('.gz'))) {
    uploadData.value.file = file
  } else {
    alert('请上传 .zip 或 .tar.gz 格式的文件')
  }
}

const doUploadPackage = async () => {
  if (!uploadData.value.file) {
    alert('请选择更新包文件')
    return
  }

  uploading.value = true
  uploadProgress.value = 0

  try {
    const token = localStorage.getItem('token')
    const formData = new FormData()
    formData.append('file', uploadData.value.file)
    formData.append('version', uploadData.value.version)
    formData.append('changelog', uploadData.value.changelog)

    const response = await fetch(`/api/skills/${uploadData.value.skillName}/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData
    })

    uploadProgress.value = 100

    const result = await response.json()
    if (result.success) {
      alert(`更新包上传成功！版本 ${uploadData.value.version} 已发布。`)
      showUploadPackage.value = false
      uploadData.value = {
        skillName: selectedSkill.value?.id || '',
        version: '',
        changelog: '',
        file: null
      }
      await loadVersions(selectedSkill.value.id)
      await loadSkills()
    } else {
      alert(result.message || '上传失败')
    }
  } catch (error) {
    console.error('Failed to upload package:', error)
    alert('上传失败: ' + error.message)
  } finally {
    uploading.value = false
    uploadProgress.value = 0
  }
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

onMounted(() => {
  loadSkills()
})
</script>
