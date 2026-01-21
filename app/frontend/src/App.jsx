
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Upload, Loader2, Maximize2, CheckCircle2, Image as ImageIcon, Download, Box, Plus } from 'lucide-react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, X, Sliders } from 'lucide-react';

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
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClickDownload(item);
            }}
            className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
            title="Download / Compress"
          >
            <Download className="w-4 h-4" />
          </button>
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

function CompressionModal({ item, onClose }) {
  const [compressionLevel, setCompressionLevel] = useState(1);
  const [alphaThreshold, setAlphaThreshold] = useState(1);
  const [shDegree, setShDegree] = useState(0);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);

  const handleExport = async () => {
    setIsExporting(true);
    setExportProgress(0);

    try {
      // GaussianSplats3D.PlyLoader.loadFromURL expects certain callbacks
      const onProgress = (percent) => setExportProgress(percent);
      const progressiveLoad = false;
      const onProgressiveLoadSectionProgress = null;
      const optimizeSplatData = true;
      const headers = null;

      const splatBuffer = await GaussianSplats3D.PlyLoader.loadFromURL(
        item.resultUrl,
        onProgress,
        progressiveLoad,
        onProgressiveLoadSectionProgress,
        alphaThreshold,
        compressionLevel,
        optimizeSplatData,
        shDegree,
        headers
      );

      const filename = item.file.name.replace(/\.[^/.]+$/, "") + ".ksplat";
      GaussianSplats3D.KSplatLoader.downloadFile(splatBuffer, filename);
      onClose();
    } catch (err) {
      console.error("Export failed:", err);
      alert("Export failed: " + err.message);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0, y: 20 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.9, opacity: 0, y: 20 }}
        className="bg-slate-800 border border-slate-700 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Settings className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h3 className="font-bold text-lg text-white">Compress & Download</h3>
              <p className="text-xs text-slate-400">Configure .ksplat export settings</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-700 rounded-full text-slate-400 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Compression Level */}
          <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
              <label className="text-slate-300 font-medium">Compression Level</label>
              <span className="text-indigo-400 font-mono bg-indigo-500/10 px-2 py-0.5 rounded">
                Level {compressionLevel}
              </span>
            </div>
            <div className="flex gap-2">
              {[0, 1, 2].map(level => (
                <button
                  key={level}
                  onClick={() => setCompressionLevel(level)}
                  className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${compressionLevel === level
                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                    : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                    }`}
                >
                  {level === 0 ? 'None' : level === 1 ? '16-bit' : '8-bit'}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 italic">
              {compressionLevel === 0 && "Lossless conversion. Large file size."}
              {compressionLevel === 1 && "Optimized 16-bit precision. Recommended."}
              {compressionLevel === 2 && "Maximum 8-bit compression. Smallest size."}
            </p>
          </div>

          {/* Alpha Threshold */}
          <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
              <label className="text-slate-300 font-medium">Alpha Threshold</label>
              <span className="text-indigo-400 font-mono bg-indigo-500/10 px-2 py-0.5 rounded">
                {alphaThreshold}/255
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="255"
              value={alphaThreshold}
              onChange={(e) => setAlphaThreshold(parseInt(e.target.value))}
              className="w-full accent-indigo-500"
            />
            <p className="text-[10px] text-slate-500 italic"> Removes splats with opacity lower than this value. Improves performance. </p>
          </div>

          {/* SH Degree */}
          <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
              <label className="text-slate-300 font-medium">Spherical Harmonics</label>
              <span className="text-indigo-400 font-mono bg-indigo-500/10 px-2 py-0.5 rounded">
                Degree {shDegree}
              </span>
            </div>
            <div className="flex gap-2">
              {[0, 1, 2].map(deg => (
                <button
                  key={deg}
                  onClick={() => setShDegree(deg)}
                  className={`flex-1 py-2 rounded-lg text-xs font-semibold border transition-all ${shDegree === deg
                    ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                    : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                    }`}
                >
                  Deg {deg}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="p-6 bg-slate-900/30 border-t border-slate-700 space-y-4">
          {isExporting ? (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Optimizing and compressing...</span>
                <span>{Math.round(exportProgress)}%</span>
              </div>
              <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-indigo-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${exportProgress}%` }}
                />
              </div>
            </div>
          ) : (
            <button
              onClick={handleExport}
              className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold shadow-lg shadow-indigo-500/20 transition-all transform hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download .ksplat
            </button>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}

function MainArea({ selectedItem, onUpload, onConvertToMesh, meshConversionStatus }) {
  if (selectedItem && selectedItem.resultUrl && selectedItem.status === 'Complete') {
    const isConverting = meshConversionStatus?.itemId === selectedItem.id && meshConversionStatus?.status === 'converting';
    const meshUrl = selectedItem.meshUrl;
    const conversionMessage = meshConversionStatus?.message || "Converting...";

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
              className={`relative overflow-hidden flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold shadow-lg transition-all transform hover:-translate-y-0.5 active:translate-y-0 ${isConverting
                ? 'bg-slate-700 text-white cursor-not-allowed border border-slate-600'
                : 'bg-violet-600 hover:bg-violet-500 text-white shadow-violet-900/30'
                }`}
            >
              {isConverting ? (
                <>
                  {/* Progress Bar Background */}
                  <div className="absolute inset-0 bg-slate-700 z-0" />
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-violet-600/50 z-0"
                    initial={{ width: 0 }}
                    animate={{ width: `${meshConversionStatus?.progress || 0}%` }}
                    transition={{ ease: "linear", duration: 0.2 }}
                  />

                  {/* Content */}
                  <div className="relative z-10 flex items-center gap-2 min-w-[140px] justify-center">
                    <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                    <span className="truncate max-w-[200px]">{conversionMessage}</span>
                    <span className="text-xs opacity-70 ml-1">{Math.round(meshConversionStatus?.progress || 0)}%</span>
                  </div>
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
  const [compressionModalItem, setCompressionModalItem] = useState(null);

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

    setMeshConversionStatus({ itemId: item.id, status: 'converting', message: 'Starting...' });

    try {
      // Extract the PLY filename from the URL
      const plyFilename = item.resultUrl.split('/').pop();

      const response = await axios.post('/api/convert-to-mesh', {
        ply_filename: plyFilename
      });

      const requestId = response.data.request_id;
      if (!requestId) {
        throw new Error(response.data.error || 'Failed to start conversion');
      }

      // Listen to SSE for mesh conversion progress
      const evtSource = new EventSource(`/api/events/${requestId}`);

      evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
          console.error("Mesh conversion event error:", data.error);
          setMeshConversionStatus({ itemId: item.id, status: 'error', error: data.error });
          alert('Mesh conversion failed: ' + data.error);
          evtSource.close();
          return;
        }

        if (data.status) {
          setMeshConversionStatus(prev => ({
            ...prev,
            message: data.status,
            progress: data.progress
          }));
        }

        if (data.mesh_url) {
          updateItemMeshUrl(item.id, data.mesh_url);
          setMeshConversionStatus({ itemId: item.id, status: 'complete' });
          evtSource.close();
        }
      };

      evtSource.onerror = (err) => {
        console.error("Mesh SSE Error", err);
        evtSource.close();

        setMeshConversionStatus(prev => {
          // If we're already complete or error, don't overwrite
          if (prev?.status === 'complete' || prev?.status === 'error') return prev;

          return {
            itemId: item.id,
            status: 'error',
            error: 'Connection lost or conversion failed'
          };
        });
      };

    } catch (err) {
      console.error('Mesh conversion error:', err);
      setMeshConversionStatus({ itemId: item.id, status: 'error', error: err.message });
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
                onClickDownload={setCompressionModalItem}
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

      <AnimatePresence>
        {compressionModalItem && (
          <CompressionModal
            item={compressionModalItem}
            onClose={() => setCompressionModalItem(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
