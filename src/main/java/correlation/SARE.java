package correlation;

import java.util.Arrays;

public class SARE implements QPPCorrelationMetric {

    class RankScore implements Comparable<RankScore> {
        int rank;
        double score;
        String qid;

        RankScore(int rank, double score) { this.rank = rank; this.score = score; this.qid = "";}
        RankScore(int rank, double score, String qid) { this.rank = rank; this.score = score; this.qid = qid;}

        @Override
        public int compareTo(RankScore o) {
            return Double.compare(this.score, o.score);
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("(").append(rank).append(", ").append(score).append(")");
            return sb.toString();
        }
    }

    @Override
    public double correlation(double[] gt, double[] pred) {
        double sAre = computeSARE(gt, pred);
        return sAre;
    }

    public double correlation(double[] gt, double[] pred, String[] qids) {
        double sAre = computeSARE(gt, pred, qids);
        return sAre;
    }

    @Override
    public String name() {
        return "SARE";
    }

    double computeSARE(double[] gt, double[] pred, String[] qids) {
        RankScore[] gt_rs = new RankScore[gt.length];
        RankScore[] pred_rs = new RankScore[pred.length];

        for (int i=0; i < gt.length; i++) {
            gt_rs[i] = new RankScore(i, gt[i], qids[i]);
            pred_rs[i] = new RankScore(i, pred[i], qids[i]);
        }

        Arrays.sort(gt_rs);
        Arrays.sort(pred_rs);

        double sare = 0;
//        System.out.printf("****PER QUERY SARE:\tQNUMBER=%d\n", qids.length);
        for (int i=0; i < gt.length; i++) {
            int rankdiff = Math.abs(gt_rs[i].rank - pred_rs[i].rank);

//            System.out.printf("****PER QUERY SARE:\tQID=%s\tDIFF:%d\n", qids[i], rankdiff);
            sare += rankdiff/(double)gt.length;
        }
        return 1.0 - sare/(double)gt.length;
    }

    double computeSARE(double[] gt, double[] pred) {
        RankScore[] gt_rs = new RankScore[gt.length];
        RankScore[] pred_rs = new RankScore[pred.length];

        for (int i=0; i < gt.length; i++) {
            gt_rs[i] = new RankScore(i, gt[i]);
            pred_rs[i] = new RankScore(i, pred[i]);
        }

        Arrays.sort(gt_rs);
        Arrays.sort(pred_rs);

        double sare = 0;
        for (int i=0; i < gt.length; i++) {
            int rankdiff = Math.abs(gt_rs[i].rank - pred_rs[i].rank);
            sare += rankdiff/(double)gt.length;
        }
        return 1.0 - sare/(double)gt.length;
    }

    public static void main(String[] args) {
        double[] gt =   {0.32, 0.15, 0.67, 0.08, 0.96, 0.45};
        double[] pred = {0.22, 0.75, 0.47, 0.83, 0.16, 0.05};

        System.out.println(String.format("SARE: %.4f", (new SARE()).correlation(gt, pred)));
    }
}
