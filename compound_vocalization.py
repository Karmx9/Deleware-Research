import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import label, find_objects
import glob
#Created a class to handle all the detection logic and to make my code more cleaner and not messier with functions
class MouseUSVDetector:
    def __init__(self, target_freq=60000, freq_tolerance=10000, 
                 amplitude_threshold=3.0, min_duration_ms=5, 
                 max_duration_ms=300, isi_threshold_ms=250,
                 chunk_duration_sec=60, min_bandwidth_hz=2000,
                 max_bandwidth_hz=20000):
        
        #Initialize the USV detector for frequency-swept vocalizations
        
        self.target_freq = target_freq
        self.freq_tolerance = freq_tolerance
        self.amplitude_threshold = amplitude_threshold
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.isi_threshold_ms = isi_threshold_ms
        self.chunk_duration_sec = chunk_duration_sec
        self.min_bandwidth_hz = min_bandwidth_hz
        self.max_bandwidth_hz = max_bandwidth_hz
        
    def load_audio(self, filepath):
        """Load WAV file"""
        try:
            sample_rate, data = wavfile.read(filepath)
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data[:, 0]
            # Normalize to float
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            return sample_rate, data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    #Again followed tutorial to create memory efficient spectrograms since it was killing my RAM in the first attempts
    def create_spectrogram_chunk(self, data, sample_rate):
        """Create spectrogram with memory-efficient parameters"""
        nperseg = int(sample_rate * 0.010) # 10ms windows
        noverlap = int(nperseg * 0.5) # 50% overlap
        
        if sample_rate > 125000:
            decimation_factor = int(sample_rate / 125000)
            data = signal.decimate(data, decimation_factor, ftype='fir')
            sample_rate = sample_rate // decimation_factor
            nperseg = int(sample_rate * 0.010)
            noverlap = int(nperseg * 0.5)
        
        f, t, Sxx = signal.spectrogram(data, sample_rate, 
                                       nperseg=nperseg, 
                                       noverlap=noverlap,
                                       scaling='spectrum')
        
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return f, t, Sxx_db, sample_rate
    # Asked AI how to set up a detection logic for frequency sweeps with a coherence check
    def detect_frequency_sweeps(self, f, t, Sxx_db):
       
        
        freq_mask = (f >= self.target_freq - self.freq_tolerance) & \
                    (f <= self.target_freq + self.freq_tolerance)
        
        freq_indices = np.where(freq_mask)[0]
        if len(freq_indices) == 0:
            return [], None, None
        
        freq_band = f[freq_mask]
        Sxx_band = Sxx_db[freq_mask, :]
        
        noise_level = np.median(Sxx_band)
        noise_std = np.std(Sxx_band[Sxx_band < np.percentile(Sxx_band, 75)])
        threshold = noise_level + self.amplitude_threshold * noise_std
        
        binary_mask = Sxx_band > threshold
        labeled_array, num_features = label(binary_mask)
        
        syllables = []
        
        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            
            time_indices = np.where(np.any(region_mask, axis=0))[0]
            freq_indices_local = np.where(np.any(region_mask, axis=1))[0]
            
            if len(time_indices) < 3 or len(freq_indices_local) == 0: # Need at least 3 time points for a sweep
                continue
            
            start_time = t[time_indices[0]]
            end_time = t[time_indices[-1]]
            duration_ms = (end_time - start_time) * 1000
            
            if not (self.min_duration_ms <= duration_ms <= self.max_duration_ms):
                continue
            
            min_freq = freq_band[freq_indices_local[0]]
            max_freq = freq_band[freq_indices_local[-1]]
            bandwidth = max_freq - min_freq
            
            if not (self.min_bandwidth_hz <= bandwidth <= self.max_bandwidth_hz):
                continue
            
            # Extraction for frequency
            contour_freqs = []
            contour_times = []
            for time_idx in time_indices:
                freq_slice = Sxx_band[:, time_idx]
                if np.max(freq_slice) > threshold:
                    peak_freq_idx = np.argmax(freq_slice)
                    contour_freqs.append(freq_band[peak_freq_idx])
                    contour_times.append(t[time_idx])

            if len(contour_freqs) < 3:
                continue

             
            try:
                correlation_matrix = np.corrcoef(contour_times, contour_freqs)
                sweep_coherence = correlation_matrix[0, 1]
            except Exception:
                sweep_coherence = 0 # Fails if contour is vertical

            if abs(sweep_coherence) < 0.3:
                continue 
            

            freq_range = np.max(contour_freqs) - np.min(contour_freqs)
            mean_freq = np.mean(contour_freqs)
            freq_change = contour_freqs[-1] - contour_freqs[0]

            if abs(freq_change) > 1000:
                sweep_type = "up" if freq_change > 0 else "down"
            else:
                sweep_type = "flat"
            
            peak_amplitude = np.max(Sxx_band[region_mask])
            
            syllables.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': duration_ms,
                'min_freq': min_freq,
                'max_freq': max_freq,
                'mean_freq': mean_freq,
                'bandwidth': bandwidth,
                'freq_range': freq_range,
                'sweep_type': sweep_type,
                'peak_amplitude': peak_amplitude,
                'contour': contour_freqs
            })
        
        return syllables, Sxx_band, threshold
    #Detection Logic for compound vocalizations
    def identify_compounds(self, syllables):
        if len(syllables) < 2:
            return []
        
        compounds = []
        current_compound = [syllables[0]]
        
        for i in range(1, len(syllables)):
            isi_ms = (syllables[i]['start_time'] - syllables[i-1]['end_time']) * 1000
            
            if isi_ms <= self.isi_threshold_ms:
                current_compound.append(syllables[i])
            else:
                if len(current_compound) >= 2:
                    compounds.append({
                        'syllables': current_compound,
                        'num_syllables': len(current_compound),
                        'start_time': current_compound[0]['start_time'],
                        'end_time': current_compound[-1]['end_time'],
                        'total_duration_ms': (current_compound[-1]['end_time'] - 
                                            current_compound[0]['start_time']) * 1000
                    })
                current_compound = [syllables[i]]
        #Adds compound if the last set of syllables meet criteria
        if len(current_compound) >= 2:
            compounds.append({
                'syllables': current_compound,
                'num_syllables': len(current_compound),
                'start_time': current_compound[0]['start_time'],
                'end_time': current_compound[-1]['end_time'],
                'total_duration_ms': (current_compound[-1]['end_time'] - 
                                    current_compound[0]['start_time']) * 1000
            })
        
        return compounds
    #Followed another youtube tutorial on how to break apart long audio files into chunks for processing since it was killing my PC RAM 
    def process_audio_in_chunks(self, data, sample_rate):
        """Process long audio files in chunks to manage memory"""
        chunk_samples = int(self.chunk_duration_sec * sample_rate)
        total_samples = len(data)
        num_chunks = int(np.ceil(total_samples / chunk_samples))
        
        all_syllables = []
        
        print(f"  Processing in {num_chunks} chunks of {self.chunk_duration_sec}s each...")
        
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * chunk_samples
            end_sample = min((chunk_idx + 1) * chunk_samples, total_samples)
            chunk_data = data[start_sample:end_sample]
            
            print(f"    Chunk {chunk_idx + 1}/{num_chunks}...", end='')
            
            f, t, Sxx_db, actual_sr = self.create_spectrogram_chunk(chunk_data, sample_rate)
            syllables, energy, threshold = self.detect_frequency_sweeps(f, t, Sxx_db)
            
            time_offset = start_sample / sample_rate
            for syl in syllables:
                syl['start_time'] += time_offset
                syl['end_time'] += time_offset
            
            all_syllables.extend(syllables)
            print(f" {len(syllables)} USV sweeps detected")
            
            del f, t, Sxx_db, chunk_data
        
        return all_syllables
    #Followed youtube tutorial and asked AI for debugging and logic checks on how to detect compound vocalizations
    def visualize_example_detections(self, filepath, data, sample_rate, all_syllables, 
                                     compounds, save_dir, num_examples=5):
        
        #Create spectrograms of example detected compounds for validation.
        
        
        if not compounds:
            print("  No compounds to visualize")
            return None
        
        compounds_by_size = sorted(compounds, key=lambda x: x['num_syllables'])
        examples = []
        if compounds_by_size: examples.append(compounds_by_size[0])
        if len(compounds_by_size) > 1: examples.append(compounds_by_size[len(compounds_by_size)//2])
        if len(compounds_by_size) > 2: examples.append(compounds_by_size[-1])
        
        print(f"  Creating spectrograms for {min(len(examples), num_examples)} example compounds...")
        
        saved_paths = []
        
        for idx, compound in enumerate(examples[:num_examples]):
            padding = 0.2 
            start_time = max(0, compound['start_time'] - padding)
            end_time = min(len(data)/sample_rate, compound['end_time'] + padding)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment = data[start_sample:end_sample]

            # First safety check 
            if segment.size == 0:
                print(f"    Skipping example {idx+1}: Audio segment is empty.")
                continue

            nperseg_viz = int(sample_rate * 0.003) 
            noverlap_viz = int(nperseg_viz * 0.9)
            segment_sr_viz = sample_rate

            if sample_rate > 125000:
                dec_factor = int(sample_rate / 125000)
                # --- SAFETY CHECK #2 ---
                if len(segment) > dec_factor * 10: 
                    segment = signal.decimate(segment, dec_factor, ftype='fir')
                    segment_sr_viz //= dec_factor
                    nperseg_viz = int(segment_sr_viz * 0.003)
                    noverlap_viz = int(nperseg_viz * 0.9)
                else:
                    print(f"    Skipping example {idx+1}: Audio segment too short for downsampling.")
                    continue
            
            # Safety Check
            if segment.size == 0:
                print(f"    Skipping example {idx+1}: Audio segment became empty after processing.")
                continue

            f, t, Sxx = signal.spectrogram(segment, segment_sr_viz, nperseg=nperseg_viz, noverlap=noverlap_viz, scaling='spectrum')
            
            # Safety check
            if Sxx.size == 0:
                print(f"    Skipping example {idx+1}: Spectrogram is empty.")
                continue

            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 9))
            freq_min_plot, freq_max_plot = (40, 80) if self.target_freq > 50000 else (20, 50)
            
            im1 = axes[0].pcolormesh(t + start_time, f/1000, Sxx_db, shading='gouraud', cmap='jet', vmin=np.percentile(Sxx_db, 20), vmax=np.percentile(Sxx_db, 99.5))
            axes[0].set_ylim([freq_min_plot, freq_max_plot])
            axes[0].set_ylabel('Frequency (kHz)')
            axes[0].set_title(f'Example Compound {idx+1}: {compound["num_syllables"]} syllables, {compound["total_duration_ms"]:.1f}ms at {compound["start_time"]:.1f}s', fontsize=13, fontweight='bold')
            
            for syl in compound['syllables']:
                axes[0].axvline(syl['start_time'], color='lime', linewidth=2, alpha=0.8)
                axes[0].axvline(syl['end_time'], color='red', linewidth=2, alpha=0.8)
                if 'contour' in syl and syl['contour']:
                    contour_times = np.linspace(syl['start_time'], syl['end_time'], len(syl['contour']))
                    axes[0].plot(contour_times, np.array(syl['contour'])/1000, 'w-', linewidth=2, alpha=0.7)
            
            plt.colorbar(im1, ax=axes[0], label='Power (dB)')
            
            zoom_range = 10
            target_khz = self.target_freq / 1000
            freq_zoom_mask = (f/1000 >= target_khz - zoom_range) & (f/1000 <= target_khz + zoom_range)
            
            #Indexing block to prevent errors in data processing
            if np.any(freq_zoom_mask):
                Sxx_zoom = Sxx_db[freq_zoom_mask, :]
                f_zoom = f[freq_zoom_mask]
                
                if Sxx_zoom.ndim == 2 and Sxx_zoom.shape[0] > 0 and Sxx_zoom.shape[1] > 0:
                    im2 = axes[1].pcolormesh(t + start_time, f_zoom/1000, Sxx_zoom, shading='gouraud', cmap='jet', vmin=np.percentile(Sxx_zoom, 20), vmax=np.percentile(Sxx_zoom, 99.5))
                    plt.colorbar(im2, ax=axes[1], label='Power (dB)')
                else:
                    axes[1].text(0.5, 0.5, 'Error plotting zoomed region', ha='center', va='center', transform=axes[1].transAxes)
            else:
                axes[1].text(0.5, 0.5, 'No data in target zoom range', ha='center', va='center', transform=axes[1].transAxes)
            
            axes[1].set_ylim([target_khz - zoom_range, target_khz + zoom_range])
            axes[1].set_ylabel('Frequency (kHz)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_title(f'Zoomed: {target_khz} kHz Region')
            
            for syl_idx, syl in enumerate(compound['syllables']):
                axes[1].axvline(syl['start_time'], color='lime', linewidth=2, alpha=0.8, label='Start' if syl_idx == 0 else "")
                axes[1].axvline(syl['end_time'], color='red', linewidth=2, alpha=0.8, label='End' if syl_idx == 0 else "")
                if 'contour' in syl and syl['contour']:
                    contour_times = np.linspace(syl['start_time'], syl['end_time'], len(syl['contour']))
                    axes[1].plot(contour_times, np.array(syl['contour'])/1000, 'w-', linewidth=2, alpha=0.9, label='Contour' if syl_idx == 0 else "")
            
            if compound['syllables']: axes[1].legend(loc='upper right')
            
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_example_{idx+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            saved_paths.append(save_path)
            print(f"    Saved example {idx+1}: {compound['num_syllables']} syllables")
        
        return saved_paths[0] if saved_paths else None
    #Followed youtube tutorial on fixing summary plot errors and how to make one 
    def create_summary_plot(self, filepath, all_syllables, compounds, save_dir, duration):
        """Create a summary visualization"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        if all_syllables:
            axes[0].scatter([s['start_time'] for s in all_syllables], [s['duration_ms'] for s in all_syllables], alpha=0.5, s=20)
            axes[0].set_title(f'{os.path.basename(filepath)} - All Detected USV Sweeps (n={len(all_syllables)})')
            axes[0].set_ylabel('Syllable Duration (ms)')
            
            axes[1].scatter([s['start_time'] for s in all_syllables], [s['mean_freq']/1000 for s in all_syllables], alpha=0.5, s=20, c='blue')
            axes[1].set_title('USV Frequency Over Time')
            axes[1].set_ylabel('Mean Frequency (kHz)')
        
        if compounds:
            axes[2].scatter([c['start_time'] for c in compounds], [c['num_syllables'] for c in compounds], alpha=0.6, s=50, c='red')
            axes[2].set_title(f'Compound Vocalizations (n={len(compounds)})')
            axes[2].set_ylabel('Number of Syllables')
        else:
            axes[2].text(0.5, 0.5, 'No compounds detected', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Compound Vocalizations (n=0)')

        if len(all_syllables) > 1:
            isis = [(all_syllables[i]['start_time'] - all_syllables[i-1]['end_time']) * 1000 for i in range(1, len(all_syllables))]
            isis_filtered = [i for i in isis if i < 2000]
            if isis_filtered:
                axes[3].hist(isis_filtered, bins=50, alpha=0.7, edgecolor='black')
                axes[3].axvline(self.isi_threshold_ms, color='red', linestyle='--', linewidth=2, label=f'Compound threshold ({self.isi_threshold_ms}ms)')
                axes[3].set_title('Distribution of Inter-Syllable Intervals')
                axes[3].set_xlabel('Inter-Syllable Interval (ms)')
                axes[3].legend()

        for ax in axes:
            ax.set_xlim([0, duration])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved summary plot: {save_path}")

    def process_folder(self, folder_path, output_dir=None):
        """Process all WAV files in folder"""
        if output_dir is None:
            output_dir = os.path.join(folder_path, "USV_Analysis_Results")
        os.makedirs(output_dir, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        if not wav_files:
            print(f"No WAV files found in {folder_path}")
            return
        
        print(f"Found {len(wav_files)} WAV files to process\n")
        all_results = []
        
        for wav_file in wav_files:
            print(f"\nProcessing: {os.path.basename(wav_file)}")
            sample_rate, data = self.load_audio(wav_file)
            if data is None: continue
            
            duration = len(data) / sample_rate
            print(f"  Sample rate: {sample_rate} Hz, Duration: {duration:.2f}s")
            
            if duration > self.chunk_duration_sec:
                all_syllables = self.process_audio_in_chunks(data, sample_rate)
            else:
                f, t, Sxx_db, _ = self.create_spectrogram_chunk(data, sample_rate)
                all_syllables, _, _ = self.detect_frequency_sweeps(f, t, Sxx_db)
            
            compounds = self.identify_compounds(all_syllables)
            
            print(f"\n  Results for {os.path.basename(wav_file)}:")
            print(f"    Total USV sweeps detected: {len(all_syllables)}")
            print(f"    Compound vocalizations found: {len(compounds)}")
            
            if all_syllables:
                up = len([s for s in all_syllables if s['sweep_type'] == 'up'])
                
                down = len([s for s in all_syllables if s['sweep_type'] == 'down'])
                
                flat = len([s for s in all_syllables if s['sweep_type'] == 'flat'])
                print(f"    Sweep types: {up} up, {down} down, {flat} flat")
            
            self.create_summary_plot(wav_file, all_syllables, compounds, output_dir, duration)
            print(f"  Creating validation spectrograms...")
            example_path = self.visualize_example_detections(wav_file, data, sample_rate, all_syllables, compounds, output_dir)
            
            if example_path and 'first_example' not in locals():
                first_example = example_path
            
            all_results.append({
                'filename': os.path.basename(wav_file),
                'duration_sec': duration,
                'num_syllables': len(all_syllables),
                'num_compounds': len(compounds),
                'compounds': compounds,
                'syllables': all_syllables
            })
        
        summary_path = os.path.join(output_dir, "detection_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Mouse USV Frequency Sweep Detection Summary\n")
            f.write("="*70 + "\n\n")
            f.write(f"Target Frequency: {self.target_freq/1000} kHz\n")
            f.write(f"Frequency Range: {(self.target_freq-self.freq_tolerance)/1000}-"
                  f"{(self.target_freq+self.freq_tolerance)/1000} kHz\n")
            f.write(f"Minimum Bandwidth: {self.min_bandwidth_hz/1000} kHz\n")
            f.write(f"ISI Threshold: {self.isi_threshold_ms} ms\n")
            f.write(f"Amplitude Threshold: {self.amplitude_threshold} std devs\n\n")
            
            for result in all_results:
                f.write(f"\n{'='*70}\n")
                f.write(f"{result['filename']}:\n")
                f.write(f"  Duration: {result['duration_sec']:.1f}s ({result['duration_sec']/60:.1f} min)\n")
                f.write(f"  Total USV sweeps: {result['num_syllables']}\n")
                if result['duration_sec'] > 0:
                    f.write(f"  Sweeps per minute: {result['num_syllables']/(result['duration_sec']/60):.1f}\n")
                f.write(f"  Compound vocalizations: {result['num_compounds']}\n")
                
                if result['syllables']:
                    up = len([s for s in result['syllables'] if s['sweep_type'] == 'up'])
                    down = len([s for s in result['syllables'] if s['sweep_type'] == 'down'])
                    flat = len([s for s in result['syllables'] if s['sweep_type'] == 'flat'])
                    f.write(f"  Sweep types: {up} up, {down} down, {flat} flat\n")
                
                if result['compounds']:
                    f.write(f"\n  Compound Details:\n")
                    for i, comp in enumerate(result['compounds'], 1):
                        f.write(f"    {i}. Time: {comp['start_time']:.2f}s, "
                                f"Syllables: {comp['num_syllables']}, "
                                f"Duration: {comp['total_duration_ms']:.1f}ms\n")
        
        print(f"\n{'='*70}")
        print(f"Analysis complete! Results saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")
        
        if 'first_example' in locals():
            print(f"\n{'='*70}")
            print("Opening example spectrogram for visual validation...")
            print(f"{'='*70}")
            try:
                from PIL import Image
                img = Image.open(first_example)
                img.show()
                print(f"\nDisplayed: {os.path.basename(first_example)}")
                print("Check if the white contour lines trace actual frequency sweeps!")
            except ImportError:
                print(f"\nPIL not available. Please manually open: {first_example}")
            except Exception as e:
                print(f"\nCouldn't display image automatically. Please open: {first_example}")
        
        return all_results

#Trying to execute between 30-90kHz range but did not work 
if __name__ == "__main__":
    print("="*70)
    print("Sweep detection ")
    print("="*70)
    
    target_freq = None
    freq_tolerance = None
    
    while True:
        print("\nWhat is the target vocalization frequency range?")
        print("  1. 20-40 kHz (Target: 30 kHz)")
        print("  2. 50-70 kHz (Target: 60 kHz)")
        print("  3. 60-80 kHz (Target: 70 kHz)")
        print("  4.  30-90 kHz (Target: 60 kHz)") 
        print("  5. Custom frequency")                     
        
        freq_choice = input("\nEnter choice (1-5) or press Enter for 60 kHz (50-70 range): ").strip()
        
        try:
            if freq_choice == "1":
                target_freq = 30000
                freq_tolerance = 10000
                freq_range = "20-40 kHz"
                break
            elif freq_choice == "3":
                target_freq = 70000
                freq_tolerance = 10000
                freq_range = "60-80 kHz"
                break
            elif freq_choice == "4": # Logic for the new option
                target_freq = 60000
                freq_tolerance = 30000 # Sets the 30-90 kHz range
                freq_range = "30-90 kHz"
                break
            elif freq_choice == "5":
                target_freq = float(input("Enter custom target frequency in kHz (e.g., 55): ")) * 1000
                freq_tolerance = float(input("Enter frequency tolerance in kHz (e.g., 5 for +/- 5kHz): ")) * 1000
                freq_range = f"{(target_freq - freq_tolerance)/1000}-{(target_freq + freq_tolerance)/1000} kHz"
                break
            elif freq_choice == "" or freq_choice == "2":
                 target_freq = 60000
                 freq_tolerance = 10000
                 freq_range = "50-70 kHz"
                 break
            else:
                 print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter numbers only for custom frequency.")

    print(f"Using target frequency: {target_freq/1000} kHz (Range: {freq_range})")

    # Execution blocks with regards to bandwidth and threshold and sensitivity
    print("\n" + "="*70)
    print("\nWhich detection sensitivity would you like?")
    print("  1. Lenient (threshold = 3.0 std) - detects more, may include noise")
    print("  2. Moderate (threshold = 4.0 std) - balanced")
    print("  3. Strict (threshold = 5.0 std) - fewer detections, higher confidence [RECOMMENDED]")
    print("  4. Custom threshold")
    
    choice = input("\nEnter choice (1-4) or press Enter for Strict: ").strip()
    
    if choice == "1":
        threshold = 3.0
        print("Using LENIENT threshold: 3.0 std")
    elif choice == "2":
        threshold = 4.0
        print("Using MODERATE threshold: 4.0 std")
    elif choice == "4":
        try:
            threshold = float(input("Enter custom threshold (e.g., 4.5): "))
            print(f"Using CUSTOM threshold: {threshold} std")
        except:
            threshold = 5.0
            print("Invalid input. Using STRICT threshold: 5.0 std")
    else: # Default is Strict
        threshold = 5.0
        print("Using STRICT threshold: 5.0 std")
    
    print("\n" + "="*70)
    print("\nFrequency sweep bandwidth (how much frequency modulation)?")
    print("  1. Standard (2-20 kHz) - typical mouse USVs [RECOMMENDED]")
    print("  2. Wide sweeps (5-20 kHz) - only large frequency changes")
    print("  3. Narrow sweeps (1-20 kHz) - include smaller frequency changes")
    
    bw_choice = input("\nEnter choice (1-3) or press Enter for Standard: ").strip()
    
    if bw_choice == "2":
        min_bw = 5000
        max_bw = 20000
        print("Using bandwidth range: 5-20 kHz (wide sweeps)")
    elif bw_choice == "3":
        min_bw = 1000
        max_bw = 20000
        print("Using bandwidth range: 1-20 kHz (narrow sweeps)")
    else: # Default is Standard
        min_bw = 2000
        max_bw = 20000
        print("Using bandwidth range: 2-20 kHz (standard)")
    
    print("\n" + "="*70 + "\n")
    
    # Initialize the detector with chosen parameters
    detector = MouseUSVDetector(
        target_freq=target_freq,
        freq_tolerance=freq_tolerance,
        amplitude_threshold=threshold,
        min_duration_ms=5,
        max_duration_ms=300,
        isi_threshold_ms=250,
        chunk_duration_sec=60,
        min_bandwidth_hz=min_bw,
        max_bandwidth_hz=max_bw
    )
    
    # Process the folder
    folder_path = r"C:\Users\sushe\OneDrive\Desktop\delawarefiles"
    results = detector.process_folder(folder_path)
    
    # Final Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nTarget frequency: {target_freq/1000} kHz ({freq_range})")
    print(f"Detection threshold: {threshold} std")
    print(f"Bandwidth range: {min_bw/1000}-{max_bw/1000} kHz")
    print("\nCheck the output folder for:")
    print("\nIf detections don't match real vocalizations, try:")
    print("   - Different bandwidth range (wider or narrower)")
    print("   - Higher/lower threshold")
    print("   - Different target frequency")
    print("="*70)
