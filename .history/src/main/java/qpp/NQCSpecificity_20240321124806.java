package qpp;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import retrieval.MsMarcoQuery;
import qrels.RetrievedResults;

import java.io.IOException;
import java.util.Arrays;

public class NQCSpecificity extends BaseIDFSpecificity {
    public NQCSpecificity(IndexSearcher searcher) {
        super(searcher);
    }

    @Override
    public double computeSpecificity(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k) {
        return computeNQC(q.getQuery(), retInfo, k);
    }

    private double computeNQC(Query q, double[] rsvs, int k) {
        //double ref = new StandardDeviation().evaluate(rsvs);
        if(rsvs.length == 0){
            return -1; //if nothing has been retrieved, return a special value telling the caller retrieval failed
        }
        double ref = Arrays.stream(rsvs).average().getAsDouble();
        double maxIDF = 0;
        double nqc = 0;
        double del;
        for (double rsv: rsvs) {
            del = rsv - ref;
            nqc += del*del;
        }
        nqc /= (double)rsvs.length;

        try {
            // dekhar jonyo je ei duto baaler modhye konta better baal!
            //avgIDF = Arrays.stream(idfs(q)).average().getAsDouble();
            maxIDF = Arrays.stream(idfs(q)).max().getAsDouble();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return nqc * maxIDF; // high variance, high avgIDF -- more specificity
    }

    public double computeNQC(Query q, RetrievedResults topDocs, int k) {
        return computeNQC(q, topDocs.getRSVs(k), k);
    }

    double[] getRSVs(TopDocs topDocs) {
        return Arrays.stream(topDocs.scoreDocs)
                .map(scoreDoc -> scoreDoc.score)
                .mapToDouble(d -> d)
                .toArray();
    }

    public double computeNQC(Query q, TopDocs topDocs, int k) {
        return computeNQC(q, getRSVs(topDocs), k);
    }

    @Override
    public String name() {
        return "nqc";
    }
}
