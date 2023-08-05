import java.awt.*;
import java.awt.image.*;
import javax.swing.JComponent;

public class VideoComponent extends JComponent {

    public void paintComponent(Graphics g) {
        // Recover Graphics2D
        Graphics2D g2 = (Graphics2D) g;
        g2.drawImage(img,0,0,this);
    }


    public void setImg(BufferedImage newimg) {
        this.img = newimg;
    }

    private BufferedImage img;
}