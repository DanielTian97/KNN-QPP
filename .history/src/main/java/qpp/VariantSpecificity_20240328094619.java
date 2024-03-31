package qpp;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import qrels.ResultTuple;
import qrels.RetrievedResults;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.TermDistribution;

import java.util.List;
import java.util.Map;

public class VariantSpecificity extends NQCSpecificity {
    QPPMethod baseModel;
    KNNRelModel knnRelModel;
    int numVariants;
    float lambda;
    double scaler; // to scale the current query's retrieval scores

    public VariantSpecificity(QPPMethod baseModel,
                              IndexSearcher searcher, KNNRelModel knnRelModel,
                              int numVariants,
                              float lambda) {
        super(searcher);

        this.baseModel = baseModel;
        this.knnRelModel = knnRelModel;
        this.numVariants = numVariants;
        this.lambda = lambda;
        this.scaler = 1;
    }

    public void setScaler(double scaler){
        this.scaler = scaler;
    }

    @Override
    public double computeSpecificity(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k) {
        List<MsMarcoQuery> knnQueries = null;
        double variantSpec = 0;

        // retInfo.getRSVs(k);

        try {
            if (numVariants > 0)
                knnQueries = knnRelModel.getKNNs(q, numVariants);

            if (knnQueries!=null && !knnQueries.isEmpty()) {
                variantSpec = variantSpecificity(q, knnQueries, retInfo, topDocs, k);
            }

        }
        catch (Exception ex) { ex.printStackTrace(); }

        return knnQueries!=null?
                lambda * variantSpec + (1-lambda) * baseModel.computeSpecificity(q, retInfo, topDocs, k) / this.scaler:
                baseModel.computeSpecificity(q, retInfo, topDocs, k);
    }

    public double preComputeRatio(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k){
        List<MsMarcoQuery> knnQueries = null;

        double variantSpec = 0;
        double currentSpec = baseModel.computeSpecificity(q, retInfo, topDocs, k);

        try {
            knnQueries = knnRelModel.getKNNs(q, 3);

            if (knnQueries!=null && !knnQueries.isEmpty()) {
                variantSpec = variantSpecificity(q, knnQueries, retInfo, topDocs, k);
            }

        }
        catch (Exception ex) { ex.printStackTrace(); }

        return knnQueries!=null ? (currentSpec / variantSpec) : -1;
    }

    double variantSpecificity(MsMarcoQuery q, List<MsMarcoQuery> knnQueries,
                              RetrievedResults retInfo, TopDocs topDocs, int k) throws Exception {
        double specScore = 0;
        double z = 0;
        double variantSpecScore;
        double refSim;

        // apply QPP base model on these estimated relevance scores
        for (MsMarcoQuery rq: knnQueries) {
            //System.out.println(rq.toString());

            TopDocs topDocsRQ = searcher.search(rq.getQuery(), k);
            RetrievedResults varInfo = new RetrievedResults(rq.getId(), topDocsRQ);
            //Arrays.stream(varInfo.getRSVs(5)).forEach(System.out::println);

            variantSpecScore = baseModel.computeSpecificity(rq, varInfo, topDocs, k);

            //if nothing has been retrieved, then set the weight to 0
            if(variantSpecScore == -1){
                refSim = 0;
            } else {
                refSim = rq.getRefSim();
            }

            //System.out.println(String.format("%s %.4f", rq.getId(), variantSpecScore));
            specScore +=  refSim * variantSpecScore ;
            z += refSim;
        }

        return z==0? baseModel.computeSpecificity(q, retInfo, topDocs, k): specScore/z;
    }

}
