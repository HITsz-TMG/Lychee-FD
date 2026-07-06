import { createApp } from 'vue';  // 从 Vue 3 中导入 createApp
import App from './App.vue'
import ElementPlus from 'element-plus'  // 导入 element-plus
import 'element-plus/dist/index.css'   // 导入 element-plus 样式
import './assets/styles.css'
import 'highlight.js/styles/atom-one-dark.css'

const app = createApp(App);  // 使用 createApp 初始化应用

app.use(ElementPlus);  // 使用 ElementPlus 插件
// app.config.globalProperties.$notify = ElementPlus.ElNotification;
app.mount('#app');  // 挂载到 #app 元素