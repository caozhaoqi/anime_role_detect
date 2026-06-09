<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
    <div class="absolute inset-0 bg-black/50 backdrop-blur-sm" @click="$emit('close')"></div>
    
    <div class="relative bg-white rounded-2xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-hidden animate-slide-up">
      <div class="bg-gradient-to-r from-primary-500 to-primary-600 px-6 py-5">
        <div class="flex items-start justify-between">
          <div>
            <h2 class="text-xl font-bold text-white">发布新技能</h2>
            <p class="text-primary-100 text-sm mt-1">分享您的技能给其他用户</p>
          </div>
          <button
            class="p-2 rounded-lg hover:bg-white/10 transition-colors text-white"
            @click="$emit('close')"
          >
            <X class="w-5 h-5" />
          </button>
        </div>
      </div>
      
      <form @submit.prevent="handleSubmit" class="p-6 overflow-y-auto max-h-[calc(90vh-140px)] scrollbar-thin">
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">技能 ID *</label>
            <input
              v-model="form.id"
              type="text"
              placeholder="ardc-your-skill"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">技能名称 *</label>
            <input
              v-model="form.name"
              type="text"
              placeholder="技能名称"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">版本号 *</label>
            <input
              v-model="form.version"
              type="text"
              placeholder="1.0.0"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">作者 *</label>
            <input
              v-model="form.author"
              type="text"
              placeholder="您的名字或邮箱"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">分类 *</label>
            <select
              v-model="form.category"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            >
              <option value="">请选择分类</option>
              <option v-for="cat in categories" :key="cat.value" :value="cat.value">
                {{ cat.label }}
              </option>
            </select>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">入口文件 *</label>
            <input
              v-model="form.entry_point"
              type="text"
              placeholder="scripts/main.py"
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
              required
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">描述</label>
            <textarea
              v-model="form.description"
              rows="3"
              placeholder="描述您的技能功能..."
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 resize-none"
            ></textarea>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">标签</label>
            <div class="flex flex-wrap gap-2 mb-2">
              <span
                v-for="(tag, index) in form.tags"
                :key="index"
                class="inline-flex items-center gap-1 px-2.5 py-1 bg-primary-50 text-primary-700 rounded-full text-sm"
              >
                {{ tag }}
                <button type="button" class="hover:text-primary-900" @click="removeTag(index)">
                  <X class="w-3.5 h-3.5" />
                </button>
              </span>
            </div>
            <div class="flex gap-2">
              <input
                v-model="newTag"
                type="text"
                placeholder="输入标签后按回车"
                class="flex-1 px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20"
                @keyup.enter="addTag"
              />
              <button
                type="button"
                class="btn btn-secondary btn-sm"
                @click="addTag"
              >
                <Plus class="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1.5">版本更新说明</label>
            <textarea
              v-model="form.release_notes"
              rows="2"
              placeholder="本次版本更新内容..."
              class="w-full px-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 resize-none"
            ></textarea>
          </div>
        </div>
      </form>
      
      <div class="border-t border-gray-100 px-6 py-4 flex items-center justify-end gap-3">
        <button class="btn btn-secondary" @click="$emit('close')">
          取消
        </button>
        <button
          type="submit"
          class="btn btn-primary flex items-center gap-2"
          @click="handleSubmit"
          :disabled="loading"
        >
          <Loader2 v-if="loading" class="w-4 h-4 animate-spin" />
          <Send v-else class="w-4 h-4" />
          {{ loading ? '发布中...' : '发布技能' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { X, Plus, Send, Loader2 } from 'lucide-vue-next'
import { skillApi } from '../api/skillApi'

const emit = defineEmits(['close', 'success'])

const loading = ref(false)
const newTag = ref('')

const form = reactive({
  id: '',
  name: '',
  version: '',
  author: '',
  category: '',
  entry_point: '',
  description: '',
  tags: [],
  release_notes: ''
})

const categories = [
  { value: 'collector', label: '数据采集' },
  { value: 'cleaner', label: '数据清洗' },
  { value: 'classifier', label: '分类识别' },
  { value: 'trainer', label: '模型训练' },
  { value: 'search', label: '搜索检索' },
  { value: 'analyzer', label: '数据分析' },
  { value: 'utility', label: '工具辅助' }
]

const addTag = () => {
  const tag = newTag.value.trim()
  if (tag && !form.tags.includes(tag) && form.tags.length < 5) {
    form.tags.push(tag)
    newTag.value = ''
  }
}

const removeTag = (index) => {
  form.tags.splice(index, 1)
}

const handleSubmit = async () => {
  if (!form.id || !form.name || !form.version || !form.author || !form.category || !form.entry_point) {
    alert('请填写必填字段')
    return
  }
  
  loading.value = true
  
  try {
    const data = {
      id: form.id,
      name: form.name,
      version: form.version,
      description: form.description,
      author: form.author,
      category: form.category,
      entry_point: form.entry_point,
      tags: form.tags,
      release_notes: form.release_notes
    }
    
    await skillApi.createSkill(data)
    emit('success')
  } catch (error) {
    console.error('Failed to create skill:', error)
    alert('发布失败，请重试')
  } finally {
    loading.value = false
  }
}
</script>
