// vue.config_dev.js - experimental build config
// Usage: vue-cli-service build --config vue.config_dev.js
const { defineConfig } = require('@vue/cli-service');

module.exports = defineConfig({
  transpileDependencies: true,
  pages: {
    index: {
      entry: 'src/main_dev.js',
      template: 'public/index_dev.html',
      filename: 'index.html',
      title: 'lychee-fd DEV',
    },
    weightTest: {
      entry: 'src/main_weight_test.js',
      template: 'public/index_dev.html',
      filename: 'weight-test.html',
      title: 'lychee-fd Weight Test',
    },
  },
  chainWebpack: (config) => {
    config.plugin('copy').tap((args) => {
      const patterns = args[0] && Array.isArray(args[0].patterns)
        ? args[0].patterns
        : [];
      for (const pattern of patterns) {
        pattern.globOptions = pattern.globOptions || {};
        const ignored = pattern.globOptions.ignore || [];
        pattern.globOptions.ignore = [
          ...ignored,
          '**/index.html',
          '**/input_cases/showcase/*#*',
        ];
      }
      return args;
    });
  },
});
