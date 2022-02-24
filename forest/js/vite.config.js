import solidPlugin from 'vite-plugin-solid';

export default {
    base: '/forest/',
    build: {
        outDir: '../templates',
        assetsDir: 'static',
        emptyOutDir: true
    },
    plugins: [solidPlugin()],
}
