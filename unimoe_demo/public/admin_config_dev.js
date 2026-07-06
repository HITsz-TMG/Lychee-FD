// admin_config_dev.js
// 实验版控制器地址。默认空 = 与页面同源 (lychee_fd.controller 同时托管前端时即可)。
// 如果你单独把控制器跑在别的端口，把它改成形如 'http://127.0.0.1:8085'
window.__UNIMOE_ADMIN_BASE__ = '';
window.__UNIMOE_ADMIN_TOKEN__ = '';
// 如果想强制覆盖后端 API base (默认走 buildGradioApiBase 同源 :7860)，可设置:
// window.__UNIMOE_GRADIO_API_BASE__ = 'http://127.0.0.1:7860';
