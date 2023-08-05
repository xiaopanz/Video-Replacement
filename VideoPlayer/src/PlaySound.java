import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.sound.sampled.DataLine.Info;

/**
 * Plays the audio file.
 * @author Xiaopan Zhang
 */
public class PlaySound implements Runnable{
	private volatile boolean running = false;
	private volatile boolean stop = false;
	private static final Object pauseLock = new Object();
	/**
	 * CONSTRUCTOR
	 */
    public PlaySound(InputStream waveStream) {
		this.waveStream = waveStream;
    }

    public void run(){
		try {
			this.play();
		}
		catch (PlayWaveException e) {
			e.printStackTrace();
			return;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public void pause() {
		synchronized (pauseLock) {
			running = false;
		}
		System.out.println("Audio Paused");
	}

	public void resume() {
		synchronized (pauseLock) {
			running = true;
			pauseLock.notifyAll(); // Unblocks thread
		}
		System.out.println("Audio Resumed");
	}

	public void stop() {
		synchronized (pauseLock) {
			running = false;
			stop = true;
			offset = dataLine.getFramePosition();
		}
		System.out.println("Audio Stopped");
	}

    /**
     * Plays the audio file.
     * @throws PlayWaveException
     */
    public void play() throws PlayWaveException, IOException {
		AudioInputStream audioInputStream = null;
		try {
			InputStream bufferedIn = new BufferedInputStream(this.waveStream);
			audioInputStream = AudioSystem.getAudioInputStream(bufferedIn);
		}
		catch (UnsupportedAudioFileException e1) {
			throw new PlayWaveException(e1);
		}
		catch (IOException e1) {
			throw new PlayWaveException(e1);
		}

		// gets the information about the AudioInputStream
		audioFormat = audioInputStream.getFormat();
		Info info = new Info(SourceDataLine.class, audioFormat);
		// opens the audio channel
		dataLine = null;
		try {
			dataLine = (SourceDataLine) AudioSystem.getLine(info);
			dataLine.open(audioFormat, 3072);
		}
		catch (LineUnavailableException e1) {
			throw new PlayWaveException(e1);
		}

		// Starts the music :P
		dataLine.start();
		int readBytes = 0;
		byte[] audioBuffer = new byte[this.EXTERNAL_BUFFER_SIZE];

		try {
			readBytes = audioInputStream.read(audioBuffer, 0, audioBuffer.length);
			//readBytes = 27366990;
			System.out.println(readBytes);
		}catch (IOException e1) {
			throw new PlayWaveException(e1);
		}
		audioInputStream.close();
		int bytesPlayed = 0;

		while (bytesPlayed < readBytes) {
			try {
				synchronized (pauseLock) {
					while (!running) {
						pauseLock.wait();
					}
				}
			} catch (InterruptedException e) {
				//do nothing just continue
			}
			if (stop) {
				bytesPlayed = 0;
				stop = false;
			}
			dataLine.write(audioBuffer, bytesPlayed, 3072);

			bytesPlayed += 3072;
//			if (bytesPlayed > readBytes) {
//				bytesPlayed = 0;
//				running = false;
//				stop = true;
//				System.out.println("Video Ended");
//			}
		}

    }

    public int getPosition() {
		return dataLine.getFramePosition() - offset;
    }

    public float getSampleRate() {
	return audioFormat.getFrameRate();
    }

	private int offset = 0;
    private SourceDataLine dataLine;
    private AudioFormat audioFormat;
    private InputStream waveStream;
	private final int  EXTERNAL_BUFFER_SIZE = 32 * 1024 * 1024 ;
	//private final int EXTERNAL_BUFFER_SIZE = 20480;
}
