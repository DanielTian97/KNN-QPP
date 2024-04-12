package qpp;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import qrels.PerQueryRelDocs;
import qrels.ResultTuple;
import retrieval.*;
import qrels.RetrievedResults;

import java.util.Arrays;
import java.util.Set;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class CoRelSpecificity extends VariantSpecificity {
    public CoRelSpecificity(QPPMethod baseModel,
                            IndexSearcher searcher, KNNRelModel knnRelModel,
                            int numVariants,
                            float lambda) {   //I deleted ', boolean normaliseScores'
        super(baseModel, searcher, knnRelModel, numVariants, lambda);   //I deleted ', boolean normaliseScores'
    }

    // @Override
    // public double computeSpecificity(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k) {
    //     List<MsMarcoQuery> knnQueries = null;
    //     double variantSpec = 0, colRelSpec = 0;
    //     double qppScore = 0;

    //     try {
    //         qppScore = (1-lambda)*baseModel.computeSpecificity(q, retInfo, topDocs, k);
    //         if (numVariants > 0)
    //             knnQueries = knnRelModel.getKNNs(q, numVariants);

    //         if (knnQueries != null) {
    //             int numRelatedQueries = knnQueries.size();
    //             colRelSpec = coRelsSpecificity(knnQueries.subList(0, Math.min(numVariants, numRelatedQueries)), k);
    //             qppScore += lambda*colRelSpec;
    //         }
    //     }
    //     catch (Exception ex) { ex.printStackTrace(); }

    //     return qppScore;
    // }

    @Override
    public double computeSpecificity(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k) {
        List<MsMarcoQuery> knnQueries = null;
        double coRelSpec = 0;

        try {
            if (numVariants > 0)
                knnQueries = knnRelModel.getKNNs(q, numVariants);

            if (knnQueries!=null && !knnQueries.isEmpty()) {
                coRelSpec = coRelsSpecificity(q, knnQueries, retInfo, topDocs, k);
                variantSpec = variantSpecificity(q, knnQueries, retInfo, topDocs, k);
            }

        }
        catch (Exception ex) { ex.printStackTrace(); }

        return knnQueries!=null?
                0.9 * lambda * variantSpec + 0.1 * lambda * coRelSpec + (1-lambda) * baseModel.computeSpecificity(q, retInfo, topDocs, k) / this.scaler:
                baseModel.computeSpecificity(q, retInfo, topDocs, k);
    }

    double coRelsSpecificity(MsMarcoQuery q, List<MsMarcoQuery> knnQueries, RetrievedResults retInfo, TopDocs topDocs, int k) throws Exception {

        int i = 1;
        double corelScore = 0, corelEstimate = 0, refSim;
        double z = 0;

        for (MsMarcoQuery rq: knnQueries) {
            PerQueryRelDocs relDocs = rq.getRelDocSet();
            if (relDocs==null || relDocs.getRelDocs().isEmpty())
                continue;
            String docName = relDocs.getRelDocs().iterator().next();
            String docText = reader.document(knnRelModel.getDocOffset(docName)).get(Constants.CONTENT_FIELD);
            MsMarcoQuery docQuery = new MsMarcoQuery(docName, docText);

            TopDocs topQueries = knnRelModel.getQueryIndexSearcher().search(docQuery.getQuery(), Constants.CLARITY_CAL_RANGE); // NOW THE NUMBER IS 5
            // System.out.println("Rel doc: " + docText);
            // for (ScoreDoc sd: topQueries.scoreDocs) {
            //     System.out.println("Retrieved query: " + knnRelModel.getQueryIndexSearcher().getIndexReader().document(sd.doc).get(Constants.CONTENT_FIELD) + ", score: " + sd.score);
            // }

            RetrievedResults topQueriesRetrievedResults = new RetrievedResults(rq.getId(), topQueries);
            // if (this.doNormalisation){
            //     topQueriesRetrievedResults = normaliseScores(topQueriesRetrievedResults);
            // }
            // topQueriesRetrievedResults = normaliseScores(topQueriesRetrievedResults); // do normalisation compulsorily

            corelEstimate = baseModel.computeSpecificity(rq, topQueriesRetrievedResults, null, Constants.CLARITY_CAL_RANGE);

            if(corelEstimate == -1){
                refSim = 0;
            } else {
                refSim = rq.getRefSim();
            }

            corelScore += refSim * corelEstimate;
            z += refSim;
        }

        return z==0? baseModel.computeSpecificity(q, retInfo, topDocs, k): corelScore/z;
    }

}