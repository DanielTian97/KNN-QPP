package correlation;

import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

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

    public double correlationWithLog(double[] gt, double[] pred, String[] qids) {
        computePerQuerySARE(gt, pred, qids, true);
        double sAre = computeSARE(gt, pred, qids);
        return sAre;
    }

//    public double correlation(double[] gt, double[] pred, String[] qids) {
//        double sAre = computeSARE(gt, pred, qids);
//        return sAre;
//    }

    @Override
    public String name() {
        return "SARE";
    }

    double computeSARE(double[] gt, double[] pred) {
        RankScore[] gt_rs = new RankScore[gt.length];
        RankScore[] pred_rs = new RankScore[pred.length];
        double[] rankDiffs = new double[pred.length];

        for (int i = 0; i < gt.length; i++) {
            gt_rs[i] = new RankScore(i, gt[i], Integer.toString(i));
            pred_rs[i] = new RankScore(i, pred[i], Integer.toString(i));
        }

        Arrays.sort(gt_rs);
        Arrays.sort(pred_rs);

        Map<String, RankScore> map_gts = new HashMap<>();
        Map<String, RankScore> map_preds = new HashMap<>();

        for (int i = 0; i < gt.length; i++) {
            gt_rs[i].rank = i;
            pred_rs[i].rank = i;
            map_gts.put(gt_rs[i].qid, gt_rs[i]);
            map_preds.put(pred_rs[i].qid, pred_rs[i]);
        }

        for (String id : map_gts.keySet()) {
            int gt_rank = map_gts.get(id).rank;
            int pred_rank = map_preds.get(id).rank;
            rankDiffs[Integer.parseInt(id)] = Math.abs(gt_rank - pred_rank) / (double) gt.length;
        }

        return Arrays.stream(rankDiffs).average().getAsDouble();
    }

    double[] computePerQuerySARE(double[] gt, double[] pred, String[] qids, boolean printLog) {
        RankScore[] gt_rs = new RankScore[gt.length];
        RankScore[] pred_rs = new RankScore[pred.length];
        double[] rankDiffs = new double[pred.length];

        for (int i = 0; i < gt.length; i++) {
            gt_rs[i] = new RankScore(i, gt[i], qids[i]);
            pred_rs[i] = new RankScore(i, pred[i], qids[i]);
        }

        Arrays.sort(gt_rs);
        Arrays.sort(pred_rs);

        Map<String, RankScore> map_gts = new HashMap<>();
        Map<String, RankScore> map_preds = new HashMap<>();

        for (int i = 0; i < gt.length; i++) {
            gt_rs[i].rank = i;
            pred_rs[i].rank = i;
            map_gts.put(gt_rs[i].qid, gt_rs[i]);
            map_preds.put(pred_rs[i].qid, pred_rs[i]);
        }

        int i = 0;
        for (String id : map_gts.keySet()) {
            int gt_rank = map_gts.get(id).rank;
            int pred_rank = map_preds.get(id).rank;
            double rankDiff = Math.abs(gt_rank - pred_rank) / (double) gt.length;
            rankDiffs[i++] = rankDiff;
            if(printLog){
                System.out.format("qid: %s, rankDiff = %.4f", id, rankDiff);
            }
        }

        return rankDiffs;
    }


    double computeSARE(double[] gt, double[] pred, String[] qids) {
        double[] rankDiffs = computePerQuerySARE(gt, pred, qids, false);
        return Arrays.stream(rankDiffs).average().getAsDouble();
    }

    public static void main(String[] args) {
        double[] gt =   {0.32, 0.15, 0.67, 0.08, 0.96, 0.45};
        double[] pred = {0.22, 0.75, 0.47, 0.83, 0.16, 0.05};
        System.out.println("Test 1:");
        System.out.println(String.format("SARE: %.4f", (new SARE()).correlation(gt, pred)));

        String[] qids =   {"100", "200", "300", "400", "500", "600"};
        System.out.println("Test 2:");
        System.out.println(String.format("SARE: %.4f", (new SARE()).correlation(gt, pred, qids)));

        System.out.println("The answers for both questions should be 0.4444.");
    }
}
