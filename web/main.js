// main.js

// Initialize the neural network module with locateFile to correctly find the .data file
NeuralNetModule({
    locateFile: function (path, prefix) {
        if (path.endsWith('.data') || path.endsWith('.wasm')) {
            return 'wasm/' + path;
        }
        return path;
    }
}).then(function (Module) {
    // Wrap the C functions to be callable from JavaScript
    const initialize_nn = Module.cwrap('initialize_nn', 'void', []);
    const classify_digit = Module.cwrap('classify_digit', 'number', ['string']);
    const cleanup_nn = Module.cwrap('cleanup_nn', 'void', []);

    // Create the Vue 3 app
    const app = Vue.createApp({
        data() {
            return {
                isDrawing: false,
                context: null,
                classification: '...',
                confidence: '...',
                modelLoaded: false,
                isSuccess: false,
                isError: false,
                points: [], // Store points for smoothing
            };
        },
        mounted() {
            this.initCanvas();
            this.loadModel();
            window.addEventListener('beforeunload', this.handleBeforeUnload);
        },
        methods: {
            initCanvas() {
                const canvas = this.$refs.canvas;
                this.context = canvas.getContext('2d');
                // Set canvas background to black
                this.context.fillStyle = "#000000";
                this.context.fillRect(0, 0, canvas.width, canvas.height);
                // Set drawing color to white
                this.context.strokeStyle = "#FFFFFF";
                this.context.lineWidth = 20; // Increase line width for better visibility when scaling down
                this.context.lineCap = "round";
                this.context.lineJoin = "round"; // Ensures smooth joints between lines
                // Enable image smoothing
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
                this.points = []; // Reset points
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
                    const lastTwoPoints = this.points.slice(-3);
                    const [point1, point2, point3] = lastTwoPoints;
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
                if (event.touches.length > 0) {
                    this.startDrawing(event.touches[0]);
                }
            },
            handleTouchMove(event) {
                event.preventDefault();
                if (event.touches.length > 0) {
                    this.draw(event.touches[0]);
                }
            },
            getCoordinates(event) {
                const canvas = this.$refs.canvas;
                const rect = canvas.getBoundingClientRect();
                let x, y;
                if (event.type.startsWith('touch')) {
                    x = event.clientX - rect.left;
                    y = event.clientY - rect.top;
                } else {
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
            async loadModel() {
                // Initialize the neural network
                initialize_nn();
                this.modelLoaded = true;
            },
            classifyDigit() {
                // Convert canvas to 28x28 grayscale image with smoothing
                const canvas = this.$refs.canvas;

                // Create a temporary canvas for preprocessing
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28;
                tempCanvas.height = 28;
                const tempCtx = tempCanvas.getContext('2d');

                // Enable image smoothing for better downscaling
                tempCtx.imageSmoothingEnabled = true;
                tempCtx.imageSmoothingQuality = 'high';

                // Apply a blur filter before scaling down
                tempCtx.filter = 'blur(1px)';

                // Draw the canvas image to the temp canvas with scaling and filtering
                tempCtx.drawImage(canvas, 0, 0, 28, 28);

                // Get image data from the temp canvas
                const imageData = tempCtx.getImageData(0, 0, 28, 28);
                const data = imageData.data;
                const pixels = [];

                for (let i = 0; i < data.length; i += 4) {
                    // Since the image is grayscale, R=G=B
                    const pixel = data[i]; // R value
                    // No inversion needed: background is 0, ink is 255
                    pixels.push(pixel);
                }

                // Convert pixels to CSV string
                const input_csv = pixels.join(',');

                console.log('Input CSV:', input_csv);

                // Call the classify_digit function from WASM
                try {
                    const result = classify_digit(input_csv);
                    if (result >= 0 && result <= 9) {
                        this.classification = 'Predicted Digit: ' + result;
                        this.confidence = '...'; // Confidence not available from WASM model
                        this.isSuccess = true;
                        this.isError = false;
                    } else {
                        this.classification = 'Classification failed. Please try again.';
                        this.confidence = '...';
                        this.isSuccess = false;
                        this.isError = true;
                    }
                } catch (error) {
                    console.error('Error during classification:', error);
                    this.classification = 'An error occurred during classification.';
                    this.confidence = '...';
                    this.isSuccess = false;
                    this.isError = true;
                }
            },
            resetCanvas() {
                this.context.fillStyle = "#000000";
                this.context.fillRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height);
                this.classification = '...';
                this.confidence = '...';
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
                cleanup_nn();
            }
        },
        beforeUnmount() {
            window.removeEventListener('beforeunload', this.handleBeforeUnload);
            cleanup_nn();
        }
    });

    // Mount the Vue app
    app.mount('#app');
}).catch(function (err) {
    console.error('Failed to initialize the NeuralNetModule:', err);
    // Optionally, display an error message to the user
    const resultElement = document.getElementById('result');
    if (resultElement) {
        resultElement.textContent = 'Failed to initialize the Neural Network module.';
        resultElement.classList.add('error');
    }
});
