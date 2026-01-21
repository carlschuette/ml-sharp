
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, Maximize2, CheckCircle2, Image as ImageIcon, Download, Box, Plus } from 'lucide-react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { motion, AnimatePresence } from 'framer-motion';

// --- Components ---

function Viewer({ url }) {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !url) return;

    // Cleanup previous viewer if exists
    if (viewerRef.current) {
      viewerRef.current.dispose();
    }

    const viewer = new GaussianSplats3D.Viewer({
      'cameraUp': [0, -1, 0],
      'initialCameraPosition': [0, 0, -5],
      'initialCameraLookAt': [0, 0, 0],
      'rootElement': containerRef.current,
      'sharedMemoryForWorkers': false,
    });

    viewerRef.current = viewer;

    // Remove zoom limits by accessing the controls
    // The viewer uses OrbitControls internally
    if (viewer.controls) {
      viewer.controls.minDistance = 0;
      viewer.controls.maxDistance = Infinity;
    }

    viewer.addSplatScene(url, {
      'splatAlphaRemovalThreshold': 5,
      'showLoadingUI': true,
      'position': [0, 0, 0],
      'rotation': [0, 0, 0, 1],
      'scale': [1.5, 1.5, 1.5]
    })
      .then(() => {
        viewer.start();
        // Also try to remove zoom limits after scene loads
        // in case controls are initialized during start
        if (viewer.controls) {
          viewer.controls.minDistance = 0;
          viewer.controls.maxDistance = Infinity;
        }
      })
      .catch(err => {
        console.error("Error loading splat scene:", err);
      });

    return () => {
      viewer.dispose();
    };
  }, [url]);

  return (
    <div ref={containerRef} className="w-full h-full" />
  );
}

function SidebarItem({ item, isSelected, onClick }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
                p-4 rounded-xl cursor-pointer border transition-all duration-200 group relative
                ${isSelected
          ? 'bg-indigo-600/20 border-indigo-500/50 shadow-lg shadow-indigo-900/20'
          : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-800 hover:border-slate-600'}
            `}
      onClick={onClick}
    >
      <div className="flex items-center gap-3 mb-2">
        <div className={`p-2 rounded-lg ${item.status === 'Complete' ? 'bg-green-500/20 text-green-400' : 'bg-slate-700 text-slate-400'}`}>
          {item.status === 'Complete' ? <Box className="w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-sm truncate text-slate-200">{item.file.name}</h4>
          <p className="text-xs text-slate-500 truncate">{item.status}</p>
        </div>
        {item.status === 'Complete' && (
          <a
            href={item.resultUrl}
            download={item.file.name.replace(/\.[^/.]+$/, "") + ".ksplat"}
            onClick={(e) => e.stopPropagation()}
            className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
            title="Download Compresed Splat (.ksplat)"
          >
            <Download className="w-4 h-4" />
          </a>
        )}
      </div>

      {/* Progress Bar */}
      {item.status !== 'Complete' && item.status !== 'Error' && (
        <div className="h-1.5 w-full bg-slate-700/50 rounded-full overflow-hidden mt-2">
          <motion.div
            className="h-full bg-indigo-500 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${item.progress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      )}
      {item.status === 'Error' && (
        <div className="text-xs text-red-400 mt-1">{item.error}</div>
      )}
    </motion.div>
  )
}

function MainArea({ selectedItem, onUpload, onConvertToMesh, meshConversionStatus }) {
  if (selectedItem && selectedItem.resultUrl && selectedItem.status === 'Complete') {
    const isConverting = meshConversionStatus?.itemId === selectedItem.id && meshConversionStatus?.status === 'converting';
    const meshUrl = selectedItem.meshUrl;

    return (
      <div className="flex-1 h-full relative bg-black">
        <Viewer url={selectedItem.resultUrl} />
        <div className="absolute top-4 left-4 pointer-events-none">
          <h2 className="text-white font-bold drop-shadow-md">{selectedItem.file.name}</h2>
        </div>

        {/* Action buttons overlay */}
        <div className="absolute bottom-6 right-6 flex gap-3 pointer-events-auto">
          {meshUrl ? (
            <a
              href={meshUrl}
              download
              className="flex items-center gap-2 px-4 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-xl font-semibold shadow-lg shadow-green-900/30 transition-all transform hover:-translate-y-0.5 active:translate-y-0"
            >
              <Download className="w-4 h-4" />
              Download Mesh
            </a>
          ) : (
            <button
              onClick={() => onConvertToMesh(selectedItem)}
              disabled={isConverting}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold shadow-lg transition-all transform hover:-translate-y-0.5 active:translate-y-0 ${isConverting
                ? 'bg-slate-600 text-slate-300 cursor-not-allowed'
                : 'bg-violet-600 hover:bg-violet-500 text-white shadow-violet-900/30'
                }`}
            >
              {isConverting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Converting...
                </>
              ) : (
                <>
                  <Box className="w-4 h-4" />
                  Convert to Mesh
                </>
              )}
            </button>
          )}
        </div>
      </div>
    );
  }

  // Upload State
  return (
    <div className="flex-1 h-full flex items-center justify-center p-8">
      <div className="max-w-xl w-full">
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-12 backdrop-blur-sm text-center">
          <div className="w-20 h-20 bg-slate-700/50 rounded-full flex items-center justify-center mx-auto mb-6">
            <Upload className="w-10 h-10 text-indigo-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Upload Images</h2>
          <p className="text-slate-400 mb-8">Select one or more images to generate 3D Splats.</p>

          <label className="inline-flex cursor-pointer">
            <input
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              onChange={onUpload}
            />
            <span className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold shadow-lg shadow-indigo-500/25 transition-all transform hover:-translate-y-0.5 active:translate-y-0">
              Select Files
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}

// --- Main App ---

function App() {
  const [queue, setQueue] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  // Handle batch file selection
  const handleUpload = async (e) => {
    if (!e.target.files) return;

    const newItems = Array.from(e.target.files).map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'Queued',
      progress: 0,
      resultUrl: null,
      error: null
    }));

    setQueue(prev => [...newItems, ...prev]); // Add to top

    // Trigger processing for each new item
    newItems.forEach(item => processItem(item));
  };

  const processItem = async (item) => {
    // Check if we should select this item (if it's the first one)
    // setSelectedId(prev => prev || item.id);

    updateItemStatus(item.id, 'Uploading...', 0);

    const formData = new FormData();
    formData.append('file', item.file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.error) throw new Error(response.data.error);

      const requestId = response.data.request_id;

      // Listen to SSE
      const evtSource = new EventSource(`/api/events/${requestId}`);

      evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
          updateItemError(item.id, data.error);
          evtSource.close();
          return;
        }

        if (data.status) updateItemStatus(item.id, data.status, data.progress);

        if (data.url) {
          updateItemComplete(item.id, data.url);
          evtSource.close();
          // Select this item when done if none selected or if user is waiting
          // setSelectedId(item.id); 
        }
      };

      evtSource.onerror = (err) => {
        console.error("SSE Error", err);
        evtSource.close();
      };

    } catch (err) {
      updateItemError(item.id, err.message);
    }
  };

  // State Helpers
  const updateItemStatus = (id, status, progress) => {
    setQueue(prev => prev.map(item => item.id === id ? { ...item, status: status || item.status, progress: progress !== undefined ? progress : item.progress } : item));
  };
  const updateItemComplete = (id, url) => {
    setQueue(prev => prev.map(item => item.id === id ? { ...item, status: 'Complete', progress: 100, resultUrl: url } : item));
  };
  const updateItemError = (id, error) => {
    setQueue(prev => prev.map(item => item.id === id ? { ...item, status: 'Error', error } : item));
  };
  const updateItemMeshUrl = (id, meshUrl) => {
    setQueue(prev => prev.map(item => item.id === id ? { ...item, meshUrl } : item));
  };

  // Mesh conversion
  const [meshConversionStatus, setMeshConversionStatus] = useState(null);

  const handleConvertToMesh = async (item) => {
    if (!item.resultUrl) return;

    setMeshConversionStatus({ itemId: item.id, status: 'converting' });

    try {
      // Extract the PLY filename from the URL
      const plyFilename = item.resultUrl.split('/').pop();

      const response = await axios.post('/api/convert-to-mesh', {
        ply_filename: plyFilename
      });

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      updateItemMeshUrl(item.id, response.data.mesh_url);
      setMeshConversionStatus({ itemId: item.id, status: 'complete' });
    } catch (err) {
      console.error('Mesh conversion error:', err);
      setMeshConversionStatus({ itemId: item.id, status: 'error', error: err.message });
      // Show error alert
      alert('Failed to convert to mesh: ' + err.message);
    }
  };

  const selectedItem = queue.find(i => i.id === selectedId);

  return (
    <div className="h-screen bg-slate-900 text-white font-sans flex overflow-hidden">

      {/* Sidebar */}
      <aside className="w-80 bg-slate-900/50 backdrop-blur border-r border-indigo-500/10 flex flex-col z-10">
        <div className="p-4 border-b border-indigo-500/10 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Maximize2 className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">Sharp Splat</h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <label className="flex items-center gap-2 w-full p-3 rounded-xl border-2 border-dashed border-slate-700 hover:border-indigo-500 hover:bg-slate-800/50 cursor-pointer transition-all group opacity-80 hover:opacity-100">
            <div className="w-8 h-8 rounded-full bg-slate-700 group-hover:bg-indigo-600 flex items-center justify-center transition-colors">
              <Plus className="w-5 h-5 text-white" />
            </div>
            <span className="font-medium text-sm text-slate-400 group-hover:text-white">New Upload</span>
            <input type="file" multiple accept="image/*" className="hidden" onChange={handleUpload} />
          </label>

          <AnimatePresence>
            {queue.map(item => (
              <SidebarItem
                key={item.id}
                item={item}
                isSelected={selectedId === item.id}
                onClick={() => setSelectedId(item.id)}
              />
            ))}
          </AnimatePresence>

          {queue.length === 0 && (
            <div className="text-center py-8 text-slate-600 text-sm">
              No generations yet.
            </div>
          )}
        </div>

        <div className="p-4 border-t border-indigo-500/10 text-xs text-slate-500 text-center">
          Powered by Apple ML-Sharp
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/20">
        <MainArea
          selectedItem={selectedItem}
          onUpload={handleUpload}
          onConvertToMesh={handleConvertToMesh}
          meshConversionStatus={meshConversionStatus}
        />
      </main>
    </div>
  );
}

export default App;
