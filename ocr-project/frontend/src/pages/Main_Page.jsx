import React, { useRef, useState, useEffect } from 'react';
import { ImageIcon, Camera, Upload, Video, X } from 'lucide-react';

// Constants
const DEVICE_TYPES = {
  MOBILE: 'mobile',
  DESKTOP: 'desktop'
};

const MOCK_PREDICTIONS = [
  { text: 'Paracetamol', accuracy: Math.floor(Math.random() * 20) + 75 },
  { text: 'Amoxicillin', accuracy: Math.floor(Math.random() * 15) + 60 },
  { text: 'Ibuprofen', accuracy: Math.floor(Math.random() * 15) + 50 }
];

// Custom Hooks
const useDeviceDetection = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => 
      /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    setIsMobile(checkMobile());
  }, []);

  return isMobile;
};

const useStreamCleanup = (stream) => {
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);
};

// Components
const ImagePreview = ({ selectedImage, onClear }) => {
  if (!selectedImage) {
    return <ImageIcon className="h-10 w-10 text-gray-400" />;
  }

  return (
    <div className="relative w-full h-full">
      <img 
        src={selectedImage} 
        alt="Preview" 
        className="w-full h-full object-contain"
      />
      <button
        onClick={onClear}
        className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600"
      >
        ×
      </button>
    </div>
  );
};

const WebcamModal = ({ isOpen, onCapture, onClose, videoRef, canvasRef }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-2xl p-4 max-w-md w-full">
        <h3 className="text-white text-lg mb-4">Take a Picture with Webcam</h3>
        <div className="relative bg-black rounded-lg mb-4 overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-64 object-cover"
          />
        </div>
        <canvas ref={canvasRef} className="hidden" />
        <div className="flex gap-2 justify-center">
          <button
            onClick={onCapture}
            className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2"
          >
            <Camera size={18} />
            Capture
          </button>
          <button
            onClick={onClose}
            className="bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold"
          >
            <X size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

const PredictionResults = ({ predictions }) => {
  if (predictions.length === 0) {
    return (
      <div className="text-gray-500 text-sm text-center py-4">
        Predictions will appear here
      </div>
    );
  }

  return predictions.map((pred, index) => (
    <div key={index} className="flex justify-between text-white text-sm mb-2">
      <span className={index === 0 ? 'text-green-300 font-semibold' : ''}>
        {pred.text}
      </span>
      <span className={getAccuracyColor(pred.accuracy)}>
        {pred.accuracy}%
      </span>
    </div>
  ));
};

const ActionButtons = ({ isMobile, onGalleryClick, onCameraClick }) => (
  <div className="flex gap-2 mb-4 justify-center">
    <button
      onClick={onGalleryClick}
      className="bg-blue-500 hover:bg-blue-600 text-white font-semibold px-4 py-2 rounded-lg text-sm flex items-center gap-2"
    >
      <Upload size={16} />
      {isMobile ? 'Gallery' : 'Upload File'}
    </button>
    
    <button
      onClick={onCameraClick}
      className="bg-purple-500 hover:bg-purple-600 text-white font-semibold px-4 py-2 rounded-lg text-sm flex items-center gap-2"
    >
      {isMobile ? <Camera size={16} /> : <Video size={16} />}
      {isMobile ? 'Camera' : 'Webcam'}
    </button>
  </div>
);

const PredictButton = ({ isLoading, disabled, onClick }) => (
  <button 
    onClick={onClick}
    disabled={disabled}
    className="bg-gradient-to-r from-orange-400 to-pink-500 text-white font-semibold px-6 py-3 rounded-xl w-full disabled:opacity-50 flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
  >
    {isLoading ? (
      <>
        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
        Analyzing...
      </>
    ) : (
      'Predict'
    )}
  </button>
);

// Utility functions
const getAccuracyColor = (accuracy) => {
  if (accuracy >= 80) return 'text-green-400';
  if (accuracy >= 60) return 'text-yellow-400';
  return 'text-orange-400';
};

const createImageUrlFromFile = (file) => URL.createObjectURL(file);

// Main Component
export default function MainPage() {
  const [predictions, setPredictions] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showWebcam, setShowWebcam] = useState(false);
  const [stream, setStream] = useState(null);
  
  const isMobile = useDeviceDetection();
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useStreamCleanup(stream);

  // Image handling
  const handleImageSelection = (file) => {
    if (file) {
      setSelectedImage(createImageUrlFromFile(file));
      setPredictions([]);
    }
  };

  const handleFileUpload = (event) => {
    handleImageSelection(event.target.files[0]);
  };

  const handleCameraCapture = (event) => {
    handleImageSelection(event.target.files[0]);
  };

  // Webcam functions
  const startWebcam = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 } 
      });
      setStream(mediaStream);
      setShowWebcam(true);
      
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      }, 100);
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Could not access webcam. Please check permissions.");
    }
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setShowWebcam(false);
  };

  const takeWebcamPicture = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      if (blob) {
        const imageUrl = createImageUrlFromFile(blob);
        setSelectedImage(imageUrl);
        setPredictions([]);
        stopWebcam();
      }
    }, 'image/jpeg', 0.8);
  };

  // Action handlers
  const openCamera = () => {
    if (isMobile) {
      cameraInputRef.current?.click();
    } else {
      startWebcam();
    }
  };

  const openGallery = () => {
    fileInputRef.current?.click();
  };

  const clearImage = () => {
    setSelectedImage(null);
    setPredictions([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  // Prediction function
  const predictImage = async () => {
    if (!selectedImage) {
      alert('Please select an image first.');
      return;
    }

    setIsLoading(true);
  
    
    setPredictions(MOCK_PREDICTIONS);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-2xl shadow-xl p-6 w-full max-w-md text-center">
        
        {/* Device Indicator */}
        <div className={`text-xs font-semibold mb-2 ${isMobile ? 'text-green-400' : 'text-blue-400'}`}>
          {isMobile ? 'Mobile Mode' : 'Desktop Mode'}
        </div>

        <h1 className="text-white text-lg font-semibold mb-4">Doctor's Handwriting Predictor</h1>

        {/* Upload Section */}
        <div className="bg-gray-700 rounded-xl p-6 mb-4">
          
          {/* Image Preview */}
          <div className="border-2 border-dashed border-gray-500 rounded-xl p-4 mb-4 w-40 h-40 mx-auto flex items-center justify-center bg-gray-600">
            <ImagePreview selectedImage={selectedImage} onClear={clearImage} />
          </div>

          {/* Hidden Inputs */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="image/*"
            className="hidden"
          />
          <input
            type="file"
            ref={cameraInputRef}
            onChange={handleCameraCapture}
            accept="image/*"
            capture="environment"
            className="hidden"
          />

          {/* Webcam Modal */}
          <WebcamModal
            isOpen={showWebcam}
            onCapture={takeWebcamPicture}
            onClose={stopWebcam}
            videoRef={videoRef}
            canvasRef={canvasRef}
          />

          {/* Action Buttons */}
          <ActionButtons
            isMobile={isMobile}
            onGalleryClick={openGallery}
            onCameraClick={openCamera}
          />

          {/* Predict Button */}
          <PredictButton
            isLoading={isLoading}
            disabled={isLoading || !selectedImage}
            onClick={predictImage}
          />
        </div>

        {/* Results Section */}
        <div className="text-left">
          <div className="flex justify-between text-gray-300 text-sm mb-2">
            <span>Prediction</span>
            <span>Confidence</span>
          </div>
          <div className="border-t border-gray-600 my-2" />

          <PredictionResults predictions={predictions} />

          <div className="border-t border-gray-600 my-3" />
          <p className="text-gray-500 text-xs text-center">
            Analyzed using CNN-BiLSTM Model
          </p>
        </div>
      </div>
    </div>
  );
}