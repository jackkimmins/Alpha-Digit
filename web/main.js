const app = Vue.createApp({
    data() {
        return {
            isLoading: true,
            isDrawing: false,
            context: null,
            classification: 'Ready!',
            modelLoaded: false,
            isSuccess: false,
            isError: false,
            points: [],
            classify_digit: null,
            cleanup_nn: null,
        };
    },
    mounted() {
        this.loadModel();
        window.addEventListener('beforeunload', this.handleBeforeUnload);
    },
    methods: {
        async loadModel() {
            try {
                const Module = await NeuralNetModule({
                    locateFile: function (path, prefix) {
                        if (path.endsWith('.data') || path.endsWith('.wasm')) return 'wasm/' + path;
                        else return path;
                    }
                });

                // These are the C functions exposed by the WASM module, please see main_wasm.cpp for the implementation 🙂
                const initialise_nn = Module.cwrap('initialise_nn', 'void', []);
                this.classify_digit = Module.cwrap('classify_digit', 'number', ['string']);
                this.cleanup_nn = Module.cwrap('cleanup_nn', 'void', []);

                initialise_nn();
                this.modelLoaded = true;
            } catch (error) {
                console.error('Failed to load the NeuralNetModule:', error);
                this.classification = 'Failed to load the neural network.';
                this.isError = true;
            } finally {
                this.isLoading = false;
                this.$nextTick(() => { this.initCanvas(); });
            }
        },
        initCanvas() {
            const canvas = this.$refs.canvas;
            if (!canvas) {
                console.error('Canvas element not found');
                return;
            }
            this.context = canvas.getContext('2d');
            this.context.fillStyle = "#000000";
            this.context.fillRect(0, 0, canvas.width, canvas.height);
            this.context.strokeStyle = "#FFFFFF";
            this.context.lineWidth = 20;
            this.context.lineCap = "round";
            this.context.lineJoin = "round";
            this.context.imageSmoothingEnabled = true;
            this.context.imageSmoothingQuality = 'high';

            // Mouse events
            canvas.addEventListener('mousedown', this.startDrawing);
            canvas.addEventListener('mousemove', this.draw);
            canvas.addEventListener('mouseup', this.stopDrawing);
            canvas.addEventListener('mouseleave', this.stopDrawing);

            // Touch events for mobile support
            canvas.addEventListener('touchstart', this.handleTouchStart, { passive: false });
            canvas.addEventListener('touchmove', this.handleTouchMove, { passive: false });
            canvas.addEventListener('touchend', this.stopDrawing, { passive: false });
            canvas.addEventListener('touchcancel', this.stopDrawing, { passive: false });
        },
        startDrawing(event) {
            this.isDrawing = true;
            this.points = [];
            const { x, y } = this.getCoordinates(event);
            this.points.push({ x, y });
            this.context.beginPath();
            this.context.moveTo(x, y);
        },
        draw(event) {
            if (!this.isDrawing) return;
            const { x, y } = this.getCoordinates(event);
            this.points.push({ x, y });

            if (this.points.length >= 3) {
                const lastThreePoints = this.points.slice(-3);
                const [point1, point2, point3] = lastThreePoints;
                const midPoint1 = this.midPoint(point1, point2);
                const midPoint2 = this.midPoint(point2, point3);

                this.context.quadraticCurveTo(point2.x, point2.y, midPoint2.x, midPoint2.y);
                this.context.stroke();
                this.context.beginPath();
                this.context.moveTo(midPoint2.x, midPoint2.y);
            }
        },
        stopDrawing() {
            if (!this.isDrawing) return;
            this.isDrawing = false;
            this.points = [];
            this.context.closePath();
        },
        handleTouchStart(event) {
            event.preventDefault();
            if (event.touches.length > 0) this.startDrawing(event.touches[0]);
        },
        handleTouchMove(event) {
            event.preventDefault();
            if (event.touches.length > 0) this.draw(event.touches[0]);
        },
        getCoordinates(event) {
            const canvas = this.$refs.canvas;
            const rect = canvas.getBoundingClientRect();
            let x, y;

            if (event instanceof Touch) {
                // Touch event
                x = event.clientX - rect.left;
                y = event.clientY - rect.top;
            } else if (event.type && event.type.startsWith('touch')) {
                // Touch event passed as event
                if (event.touches.length > 0) {
                    x = event.touches[0].clientX - rect.left;
                    y = event.touches[0].clientY - rect.top;
                } else if (event.changedTouches.length > 0) {
                    x = event.changedTouches[0].clientX - rect.left;
                    y = event.changedTouches[0].clientY - rect.top;
                }
            } else {
                // Mouse event
                x = event.offsetX;
                y = event.offsetY;
            }

            return { x, y };
        },
        midPoint(p1, p2) {
            return {
                x: (p1.x + p2.x) / 2,
                y: (p1.y + p2.y) / 2
            };
        },
        classifyDigit() {
            if (!this.modelLoaded) {
                this.classification = 'Model not loaded yet.';
                this.isError = true;
                return;
            }

            // We need to downscale the canvas image to match the input size of the model (28x28)
            const canvas = this.$refs.canvas;

            // Temporary canvas for preprocessing
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');

            // Found that this leads to better results as it matches the original MNIST CSV dataset better
            tempCtx.imageSmoothingEnabled = true;
            tempCtx.imageSmoothingQuality = 'high';
            tempCtx.filter = 'blur(1px)';

            tempCtx.drawImage(canvas, 0, 0, 28, 28);

            // Get image data from the temp canvas
            const imageData = tempCtx.getImageData(0, 0, 28, 28);
            const data = imageData.data;
            const pixels = [];

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const grayscale = 0.21 * r + 0.72 * g + 0.07 * b;
                pixels.push(Math.round(grayscale));
            }

            // Convert pixels to CSV string
            const input_csv = pixels.join(',');

            console.log('Input CSV:', input_csv);

            try {
                const result = this.classify_digit(input_csv);
                if (result >= 0 && result <= 9) {
                    this.classification = 'Predicted Digit: ' + result;
                    this.isSuccess = true;
                    this.isError = false;
                } else {
                    this.classification = 'Classification failed. Please try again.';
                    this.isSuccess = false;
                    this.isError = true;
                }
            } catch (error) {
                console.error('Error during classification:', error);
                this.classification = 'An error occurred during classification.';
                this.isSuccess = false;
                this.isError = true;
            }
        },
        resetCanvas() {
            this.context.fillStyle = "#000000";
            this.context.fillRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height);
            this.classification = 'Ready!';
            this.isSuccess = false;
            this.isError = false;
        },
        downloadDigit() {
            const canvas = this.$refs.canvas;
            const link = document.createElement('a');
            link.href = canvas.toDataURL('image/png');
            link.download = 'digit.png';
            link.click();
        },
        handleBeforeUnload() {
            if (this.cleanup_nn) {
                this.cleanup_nn();
            }
        }
    },
    beforeUnmount() {
        window.removeEventListener('beforeunload', this.handleBeforeUnload);
        if (this.cleanup_nn) {
            this.cleanup_nn();
        }
    }
});

app.mount('#app');