// service-worker.js
self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open('AlphaDigit-store').then((cache) => cache.addAll([
            '/',
            '/index.html',
            '/style.css',
            '/main.js',
            '/manifest.json',
            '/wasm/NeuralNetModule.data',
            '/wasm/NeuralNetModule.js',
            '/wasm/NeuralNetModule.wasm'
        ])),
    );
});

self.addEventListener('fetch', (e) => {
    e.respondWith(caches.match(e.request).then((response) => response || fetch(e.request)), );
});  