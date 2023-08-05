import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class Main {
    public static void main(String[] args) {
        try {
            if (args.length < 2) {
                System.err.println("Usage: java Main video.rgb audio.wav");
                return;
            }
            String rgbFilename = args[0];
            String wavFilename = args[1];

            // opens the inputStream
            FileInputStream inputStream = new FileInputStream(wavFilename);


            PlaySound playSound = new PlaySound(inputStream);
            PlayRgbVideo imageReader = new PlayRgbVideo(rgbFilename, playSound);

            // Create two threads running video and audio
            Thread audio = new Thread(playSound);
            Thread video = new Thread(imageReader);

            audio.start();
            video.start();
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}