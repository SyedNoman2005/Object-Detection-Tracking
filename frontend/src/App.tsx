import React, { useState } from 'react';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'live'>('upload');

  return (
    <div className="min-h-screen p-8">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          AI Surveillance Dashboard
        </h1>
        <p className="text-gray-400 mt-2">Real-Time YOLOv8 Object Detection & Tracking</p>
      </header>

      <div className="max-w-6xl mx-auto bg-gray-800 bg-opacity-50 backdrop-blur-md rounded-2xl p-6 shadow-2xl border border-gray-700">
        <nav className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-2 rounded-full font-semibold transition-all ${
              activeTab === 'upload' ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Upload Video
          </button>
          <button
            onClick={() => setActiveTab('live')}
            className={`px-6 py-2 rounded-full font-semibold transition-all ${
              activeTab === 'live' ? 'bg-emerald-500 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Live Webcam
          </button>
        </nav>

        <main>
          {activeTab === 'upload' && (
            <div className="text-center p-12 border-2 border-dashed border-gray-600 rounded-xl">
              <h2 className="text-2xl mb-4">Upload Video</h2>
              <p className="text-gray-400 mb-6">Drag and drop an MP4, MOV, or AVI file here.</p>
              <input type="file" className="hidden" id="video-upload" />
              <label htmlFor="video-upload" className="cursor-pointer bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg font-semibold transition-colors">
                Select File
              </label>
            </div>
          )}
          {activeTab === 'live' && (
            <div className="text-center p-12">
              <h2 className="text-2xl mb-4">Live Webcam Tracking</h2>
              <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-inner flex items-center justify-center">
                <p className="text-gray-500">Camera stream will appear here.</p>
              </div>
              <button className="mt-6 bg-emerald-600 hover:bg-emerald-700 px-8 py-3 rounded-lg font-semibold transition-colors">
                Start Tracking
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
