    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const fileQueue = document.getElementById('fileQueue');
    const fileList = document.getElementById('fileList');
    const queueCount = document.getElementById('queueCount');
    const clearBtn = document.getElementById('clearBtn');
    const uploadBtn = document.getElementById('uploadBtn');

    const resultsContainer = document.getElementById('resultsContainer');
    const detectionCanvas = document.getElementById('detectionCanvas');
    const detectionCountBadge = document.getElementById('detectionCountBadge');
    const detectionList = document.getElementById('detectionList');

    let filesArray = [];

    const CLASS_MAPPING = {
      1: "fishnet",
      8: "pipe",
      9: "cylinder",
      "bottle": "fishnet",
      "shampoo-bottle": "pipe",
      "standing-bottle": "cylinder"
    };

    const formatFileSize = (bytes) => {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
    };

    const updateUI = () => {
      if (filesArray.length === 0) {
        fileQueue.classList.add('hidden');
        uploadBtn.disabled = true;
        uploadBtn.className = "w-full sm:w-auto font-mono text-xs uppercase tracking-[0.2em] px-8 py-3.5 transition-all duration-200 bg-[#11202a] text-[#435d6c] cursor-not-allowed border border-[#172b39]";
      } else {
        fileQueue.classList.remove('hidden');
        queueCount.textContent = `Queued Datasets (${filesArray.length})`;
        uploadBtn.disabled = false;
        uploadBtn.className = "w-full sm:w-auto font-mono text-xs uppercase tracking-[0.2em] px-8 py-3.5 transition-all duration-200 bg-[#5eead4] text-[#070e14] hover:bg-[#4dd0bc] font-medium cursor-pointer";

        fileList.innerHTML = '';
        filesArray.forEach((file, idx) => {
          const row = document.createElement('div');
          row.className = "px-5 py-3 flex items-center justify-between text-xs font-mono";
          row.innerHTML = `
            <div class="flex items-center space-x-3 truncate">
              <span class="text-[#5eead4] text-[10px] border border-[#5eead4]/30 px-1.5 py-0.5">
                ${file.type}
              </span>
              <span class="text-[#c7d7e0] truncate">${file.name}</span>
              <span class="text-[#4b6877] text-[11px]">(${file.size})</span>
            </div>
            <button onclick="removeFile(${idx})" class="text-[#4b6877] hover:text-white pl-4">✕</button>
          `;
          fileList.appendChild(row);
        });
      }
    };

    const handleFiles = (files) => {
      Array.from(files).forEach((f) => {
        filesArray.push({
          rawFile: f,
          name: f.name,
          size: formatFileSize(f.size),
          type: f.name.split('.').pop()?.toUpperCase() || 'RAW'
        });
      });
      updateUI();
    };

    window.removeFile = (index) => {
      filesArray.splice(index, 1);
      updateUI();
    };

    clearBtn.addEventListener('click', () => {
      filesArray = [];
      updateUI();
      if (resultsContainer) resultsContainer.classList.add('hidden');
    });

    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFiles(e.target.files);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('border-[#5eead4]', 'bg-[#0c1822]');
        dropZone.classList.remove('border-[#142733]', 'bg-[#09131c]/60');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('border-[#5eead4]', 'bg-[#0c1822]');
        dropZone.classList.add('border-[#142733]', 'bg-[#09131c]/60');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    });

    const renderSingleImageCanvas = (canvasEl, fileObj, detections) => {
      const reader = new FileReader();
      reader.onload = (evt) => {
        const img = new Image();
        img.onload = () => {
          canvasEl.width = img.width;
          canvasEl.height = img.height;
          const ctx = canvasEl.getContext('2d');
          
          ctx.drawImage(img, 0, 0);

          detections.forEach((det) => {
            const { x1, y1, x2, y2 } = det.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            const rawLabel = det.class_name || 'anomaly';
            const label = (CLASS_MAPPING[rawLabel] || CLASS_MAPPING[det.class_id] || rawLabel).toUpperCase();
            const scorePct = Math.round((det.confidence || 0) * 100);

            // Bounding box style
            ctx.strokeStyle = '#4FD6B5';
            ctx.lineWidth = Math.max(2, Math.round(img.width / 400));
            ctx.strokeRect(x1, y1, width, height);

            // Corner ticks
            ctx.fillStyle = '#5eead4';
            const tickLen = Math.min(width, height) * 0.15;
            ctx.fillRect(x1, y1, tickLen, 3);
            ctx.fillRect(x1, y1, 3, tickLen);
            ctx.fillRect(x2 - tickLen, y1, tickLen, 3);
            ctx.fillRect(x2 - 3, y1, 3, tickLen);

            // Label background
            const fontHeight = Math.max(12, Math.round(img.height / 35));
            ctx.font = `500 ${fontHeight}px "JetBrains Mono", monospace`;
            const textStr = `${label} [${scorePct}%]`;
            const textWidth = ctx.measureText(textStr).width;
            
            ctx.fillStyle = '#0B1A24';
            ctx.fillRect(x1, Math.max(0, y1 - fontHeight - 6), textWidth + 10, fontHeight + 6);

            // Label border
            ctx.strokeStyle = '#4FD6B5';
            ctx.lineWidth = 1;
            ctx.strokeRect(x1, Math.max(0, y1 - fontHeight - 6), textWidth + 10, fontHeight + 6);

            // Label text
            ctx.fillStyle = '#5eead4';
            ctx.fillText(textStr, x1 + 5, Math.max(fontHeight, y1 - 4));
          });
        };
        img.src = evt.target.result;
      };
      reader.readAsDataURL(fileObj);
    };

    uploadBtn.addEventListener('click', async () => {
      if (filesArray.length === 0) return;

      uploadBtn.disabled = true;
      uploadBtn.textContent = `Analyzing ${filesArray.length} Sonar Image(s)...`;
      uploadBtn.className = "w-full sm:w-auto font-mono text-xs uppercase tracking-[0.2em] px-8 py-3.5 transition-all duration-200 bg-[#122834] text-[#5eead4] border border-[#5eead4]/40 animate-pulse cursor-wait";

      try {
        const multiResultsList = document.getElementById('multiResultsList');
        if (multiResultsList) multiResultsList.innerHTML = '';
        if (resultsContainer) resultsContainer.classList.remove('hidden');

        let totalTargetCount = 0;

        for (let i = 0; i < filesArray.length; i++) {
          const item = filesArray[i];
          const formData = new FormData();
          formData.append('file', item.rawFile);

          const apiUrl = '/analysis/test-image?model_type=sss&confidence=0.10';
          const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
          });

          if (!response.ok) {
            console.error(`Error processing file ${item.name}: status ${response.status}`);
            continue;
          }

          const data = await response.json();
          const rawDetections = data.detections || [];

          // Map bottle-related classes to fishnet, pipe, cylinder
          const detections = rawDetections.map((det) => {
            const cName = det.class_name;
            const cId = det.class_id;
            const mappedName = CLASS_MAPPING[cId] || CLASS_MAPPING[cName] || cName;
            return {
              ...det,
              class_name: mappedName
            };
          });

          totalTargetCount += detections.length;

          // Create container card for this image
          const card = document.createElement('div');
          card.className = "border border-[#142733] bg-[#050b10] p-5 rounded space-y-4";

          const cardHeader = document.createElement('div');
          cardHeader.className = "flex items-center justify-between border-b border-[#142733] pb-3";
          cardHeader.innerHTML = `
            <div class="flex items-center space-x-2">
              <span class="text-xs text-[#5eead4] font-bold uppercase tracking-wider">File [${i + 1}/${filesArray.length}]: ${item.name}</span>
            </div>
            <span class="text-[11px] text-[#7d99a6] font-mono border border-[#142733] px-2.5 py-0.5">
              Target Anomalies: ${detections.length}
            </span>
          `;

          const canvasWrap = document.createElement('div');
          canvasWrap.className = "w-full flex justify-center bg-[#09131c] border border-[#142733] p-3 overflow-hidden";
          
          const canvasEl = document.createElement('canvas');
          canvasEl.className = "max-w-full h-auto rounded border border-[#172b39]";
          canvasWrap.appendChild(canvasEl);

          const targetListWrap = document.createElement('div');
          targetListWrap.className = "w-full";
          targetListWrap.innerHTML = `<h4 class="text-xs text-[#8ea7b6] uppercase tracking-widest mb-2">Detected Targets</h4>`;
          
          const targetList = document.createElement('div');
          targetList.className = "divide-y divide-[#142733] border border-[#142733] bg-[#09131c]";

          if (detections.length === 0) {
            targetList.innerHTML = `<div class="p-3 text-xs text-[#7d99a6]">No targets detected above confidence threshold.</div>`;
          } else {
            detections.forEach((det, idx) => {
              const row = document.createElement('div');
              row.className = "p-3 flex flex-col sm:flex-row sm:items-center justify-between text-xs font-mono gap-2";
              const scorePct = Math.round(det.confidence * 100);
              const bboxStr = `[${det.bbox.x1}, ${det.bbox.y1}, ${det.bbox.x2}, ${det.bbox.y2}]`;

              row.innerHTML = `
                <div class="flex items-center space-x-3">
                  <span class="text-[#5eead4] font-bold">#${idx + 1}</span>
                  <div>
                    <div class="text-[#c7d7e0] uppercase font-semibold text-xs">${det.class_name}</div>
                    <div class="text-[#698a9c] text-[10px] mt-0.5">BBOX: ${bboxStr}</div>
                  </div>
                </div>
                <div class="flex items-center space-x-3">
                  <span class="text-[#5eead4] border border-[#5eead4]/40 px-2 py-0.5 text-[10px]">
                    ${scorePct}% CONFIDENCE
                  </span>
                  <span class="text-[9px] text-[#5eead4] bg-[#5eead4]/10 border border-[#5eead4]/30 px-2 py-0.5 uppercase tracking-wider">
                    CONFIRMED ANOMALY
                  </span>
                </div>
              `;
              targetList.appendChild(row);
            });
          }

          targetListWrap.appendChild(targetList);

          card.appendChild(cardHeader);
          card.appendChild(canvasWrap);
          card.appendChild(targetListWrap);

          multiResultsList.appendChild(card);

          // Render canvas bounding boxes
          renderSingleImageCanvas(canvasEl, item.rawFile, detections);
        }

        if (detectionCountBadge) {
          detectionCountBadge.textContent = `Total Target Anomalies Across ${filesArray.length} Image(s): ${totalTargetCount}`;
        }

      } catch (err) {
        console.error("Multi-image analysis failed:", err);
        alert("Error executing SSS sonar detection for uploaded images.");
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload Data";
        uploadBtn.className = "w-full sm:w-auto font-mono text-xs uppercase tracking-[0.2em] px-8 py-3.5 transition-all duration-200 bg-[#5eead4] text-[#070e14] hover:bg-[#4dd0bc] font-medium cursor-pointer";
      }
    });