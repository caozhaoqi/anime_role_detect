import os
import json
import requests
import uuid

app_id = os.environ.get('FEISHU_APP_ID')
app_secret = os.environ.get('FEISHU_APP_SECRET')

# 获取access token
token_url = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
token_response = requests.post(token_url, headers={'Content-Type': 'application/json'}, json={'app_id': app_id, 'app_secret': app_secret}, timeout=10)
token_data = token_response.json()
print(f'Token response: {token_data}')

access_token = token_data.get('tenant_access_token')
print(f'Access token: {access_token[:20]}...')

# 发送消息 - 根据用户之前成功的格式
msg_url = 'https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id'
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

content = '测试飞书通知 - 数据采集中 (修复后)'
data = {
    'receive_id': 'oc_b376c0f5a01eef8f6240b1f3f7b249d2',
    'msg_type': 'text',
    'content': json.dumps({'text': content}),
    'uuid': str(uuid.uuid4())
}

print(f'Request data: {json.dumps(data, ensure_ascii=False)}')

response = requests.post(msg_url, headers=headers, json=data, timeout=10)
result = response.json()
print(f'Response: {result}')