# 🎓 University Gate AI - Face Recognition System

A real-time face recognition attendance system for university gates, built with Python, OpenCV, MTCNN, FaceNet, and FAISS for high-performance face detection and recognition.

## 🌟 Features

- **Real-time Face Detection**: MTCNN-based face detection with high accuracy
- **Fast Face Recognition**: FaceNet embeddings with FAISS for sub-millisecond similarity search
- **Multithreaded Camera Feed**: Optimized camera capture with separate processing threads
- **Scalable Database**: FAISS-powered database supporting thousands of identities
- **Attendance Logging**: Automatic attendance tracking with CSV export
- **Quality Validation**: Face quality assessment for reliable registration
- **Multiple Registration Modes**: Camera, single image, or batch registration
- **Real-time UI**: Live preview with performance statistics and attendance display

## 🏗️ Project Structure

```
university_gate_ai/
├── camera_feed/
│   └── capture_feed.py            # Multithreaded webcam stream
├── face_recognition/
│   ├── detector.py                # Face detection logic (MTCNN)
│   ├── recognizer.py              # Face recognition with FAISS
│   └── face_utils.py              # Embedding extraction helpers
├── database/
│   └── student_faces.npy          # Face embeddings database
├── models/                        # Model files (auto-downloaded)
├── logs/                          # Attendance logs (auto-created)
├── main.py                        # Main application entry point
├── register_faces.py              # Face registration script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd university_gate_ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Register Faces

Before running the main system, register some faces in the database:

```bash
# Register from camera (interactive)
python register_faces.py --mode camera --name "John Doe"

# Register from image file
python register_faces.py --mode image --image "path/to/photo.jpg" --name "Jane Smith"

# Batch register from directory (uses filenames as names)
python register_faces.py --mode batch --directory "path/to/photos/"
```

### 3. Run the System

```bash
# Run with default settings
python main.py

# Run with custom settings
python main.py --camera 0 --confidence 0.9 --similarity 0.6

# Run without preview (headless mode)
python main.py --no-preview
```

## 📋 Command Line Options

### Main Application (`main.py`)

```bash
python main.py [OPTIONS]

Options:
  --camera INT        Camera device ID (default: 0)
  --confidence FLOAT  Face detection confidence threshold (default: 0.9)
  --similarity FLOAT  Face recognition similarity threshold (default: 0.6)
  --no-preview       Disable live preview window
  --no-logs          Disable attendance logging
```

### Face Registration (`register_faces.py`)

```bash
python register_faces.py --mode MODE [OPTIONS]

Modes:
  camera    Register from live camera feed
  image     Register from single image file
  batch     Register from directory of images

Options:
  --image PATH       Path to image file (for image mode)
  --directory PATH   Path to directory (for batch mode)
  --name NAME        Person name (for camera/image mode)
  --count INT        Number of captures (for camera mode, default: 1)
  --confidence FLOAT Detection confidence threshold (default: 0.9)
  --min-size INT     Minimum face size (default: 80)
  --no-validation    Skip quality validation
  --report PATH      Save registration report to file
```

## 🎮 Interactive Controls

### Main Application
- **q**: Quit the application
- **s**: Show detailed system statistics
- **r**: Register new face from current frame
- **c**: Clear attendance log

### Face Registration (Camera Mode)
- **SPACE**: Capture face for registration
- **q**: Quit registration
- **r**: Restart current capture

## 📊 Performance Features

### Real-time Statistics
- Live FPS counter
- Face detection count
- Face recognition count
- Processing time metrics
- Database search performance

### Attendance Logging
- Automatic CSV log generation
- Duplicate prevention (5-second cooldown)
- Daily log files in `logs/` directory
- Real-time attendance display

## 🔧 Technical Details

### Face Detection (MTCNN)
- Multi-task CNN for robust face detection
- Handles various lighting conditions and angles
- Configurable confidence thresholds
- Facial keypoint detection

### Face Recognition (FaceNet + FAISS)
- 512-dimensional face embeddings
- Cosine similarity matching
- FAISS index for fast similarity search
- Sub-millisecond search times
- Supports thousands of identities

### Camera System
- Multithreaded frame capture
- Optimized buffer management
- Configurable resolution and FPS
- Minimal latency design

## 📈 Scalability

The system is designed to scale efficiently:

- **Database Size**: Supports 10,000+ identities with FAISS
- **Search Speed**: Sub-millisecond face matching
- **Memory Usage**: Optimized embedding storage
- **Processing**: Real-time performance on CPU
- **Future GPU Support**: Ready for CUDA acceleration

## 🔮 Future Integrations

The modular design supports easy integration with:

- **Web Dashboard**: Flask/Streamlit interface
- **Notifications**: SMS/Email alerts for attendance
- **Database**: PostgreSQL/MySQL integration
- **API**: REST API for external systems
- **Mobile App**: React Native companion app
- **Analytics**: Advanced attendance analytics

## 📝 File Formats

### Database Files
- `database/student_faces.npy`: NumPy format with embeddings and metadata
- `database/faiss_index.index`: FAISS index file for fast search

### Log Files
- `logs/attendance_YYYYMMDD.csv`: Daily attendance logs
- Format: `timestamp,name,similarity,status`

### Registration Reports
- JSON format with detailed registration statistics
- Includes success/failure analysis and quality metrics

## 🛠️ Configuration

### Model Settings
- Face detection confidence: 0.9 (adjustable)
- Face recognition threshold: 0.6 (adjustable)
- Minimum face size: 40px (adjustable)
- Embedding dimension: 512 (fixed)

### Performance Settings
- Camera buffer size: 2 frames
- FPS target: 30 (auto-adjusted)
- Processing thread: Single (expandable)

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Try different camera IDs
   python main.py --camera 1
   ```

2. **Low detection accuracy**
   ```bash
   # Lower confidence threshold
   python main.py --confidence 0.8
   ```

3. **Slow performance**
   ```bash
   # Reduce camera resolution or use GPU
   # Edit capture_feed.py to change resolution
   ```

4. **Memory issues**
   ```bash
   # Clear old logs and reduce buffer size
   # Check available RAM for large databases
   ```

### Dependencies Issues

If you encounter issues with specific packages:

```bash
# For OpenCV issues
pip install opencv-python-headless

# For PyTorch issues (CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For FAISS issues
pip install faiss-cpu --no-cache
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MTCNN**: Multi-task CNN for face detection
- **FaceNet**: Face recognition embeddings
- **FAISS**: Efficient similarity search
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

## 📞 Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

---

**Built with ❤️ for university security and attendance management**