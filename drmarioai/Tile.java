package drmarioai;

public interface Tile {
  public int EMPTY = 0x00;
  public int LEFT = 0x01;
  public int RIGHT = 0x02;
  public int TOP = 0x03;
  public int BOTTOM = 0x04;
  public int SQUARE = 0x05;
  public int VIRUS = 0x06;
  
  public int TILE_MASK = 0x0F;  
}
