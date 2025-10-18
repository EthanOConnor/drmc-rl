package drmarioai;

public class SearchChain {

  private final Searcher searcher1 = new Searcher();
  private final Searcher searcher2 = new Searcher();
  private final Searcher searcher3 = new Searcher();
  
  private final Evaluator evaluator = new DefaultEvaluator();
  
  private int currentColor1;
  private int currentColor2;
  private int nextColor1;
  private int nextColor2;
  
  private int x1;
  private int y1;
  private int o1;
  
  private int bestX;
  private int bestY;
  private int bestOrientation;
  
  private double bestValue;

  public SearchChain() {
    
    searcher1.setTargetListener((x, y, o) -> {
      this.x1 = x;
      this.y1 = y;
      this.o1 = o;
      searcher2.setPlayfield(searcher1.getPlayfield());
      searcher2.lockPill(x, y, o, currentColor1, currentColor2);
      if (searcher2.canSpawn()) {
        searcher2.search();
      }
    });
    
    searcher2.setTargetListener((x, y, o) -> {
      searcher3.setPlayfield(searcher2.getPlayfield());
      searcher3.lockPill(x, y, o, nextColor1, nextColor2);
      final double value = evaluator.evaluate(searcher3.getPlayfield());
      if (value > bestValue) {
        bestValue = value;
        bestX = x1;
        bestY = y1;
        bestOrientation = o1;
      }
    });
  }
  
  public boolean search(final int[][] playfield, 
      final int currentColor1, final int currentColor2,
      final int nextColor1, final int nextColor2) {
    
    this.searcher1.setPlayfield(playfield);
    
    this.currentColor1 = currentColor1;
    this.currentColor2 = currentColor2;
    
    this.nextColor1 = nextColor1;
    this.nextColor2 = nextColor2;
    
    this.bestValue = -Double.MAX_VALUE;
    
    if (searcher1.canSpawn()) {
      searcher1.search();
    }
    
    return bestValue != -Double.MAX_VALUE;
  }
  
  public int getMoves(final int[] list) {
    return searcher1.getMoves(bestX, bestY, bestOrientation, list);
  }
  
  public int getX() {
    return bestX;
  }
  
  public int getY() {
    return bestY;
  }
  
  public int getOrientation() {
    return bestOrientation;
  }
}
