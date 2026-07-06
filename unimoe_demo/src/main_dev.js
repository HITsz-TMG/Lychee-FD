import { createApp } from 'vue';
import App from './App_dev.vue';
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import './assets/styles.css';
import 'highlight.js/styles/atom-one-dark.css';

const app = createApp(App);
app.use(ElementPlus);
app.mount('#app');
