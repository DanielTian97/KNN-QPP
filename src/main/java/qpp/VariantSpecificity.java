package qpp;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import qrels.ResultTuple;
import qrels.RetrievedResults;
import qrels.AllRetrievedResults; // for M=M' test
import retrieval.Constants;
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
    double scaler; // to scale the current query's retrieval scores; THIS IDEA IS CEASED TO USE.
    boolean doNormalisation; //temporarily hard coded
    AllRetrievedResults qvResults;

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
//        this.doNormalisation = true; // changed to true temporarily - 0429 - 0514
        this.doNormalisation = false; // changed back to false - 0430.15:15
        this.qvResults = null;
    }

    public void setQvResults(AllRetrievedResults savedQvResults){
        this.qvResults = savedQvResults;
    }

    public void setScaler(double scaler){
        this.scaler = scaler;
    }

    RetrievedResults normaliseScores(RetrievedResults retInfo) {
        double minScore = retInfo.getTuples()
                .stream().map(x->x.getScore()).reduce(Double::min).get();
        double maxScore = retInfo.getTuples()
                .stream().map(x->x.getScore()).reduce(Double::max).get();
        double diff = maxScore - minScore;

        if (this.doNormalisation) {
            retInfo.getTuples()
                    .forEach(
                            x -> x.setScore((x.getScore()-minScore)/diff)
                    );
        }
        return retInfo;
    }

    @Override
    public double computeSpecificity(MsMarcoQuery q, RetrievedResults retInfo, TopDocs topDocs, int k, boolean verbose) {
        List<MsMarcoQuery> knnQueries = null;
        double variantSpec = 0;

        if(verbose) {
            System.out.printf("target qid: %s, lambda=%f \n", q.getId(), lambda);
        }

        // retInfo.getRSVs(k);
        if(this.doNormalisation) {
            retInfo = normaliseScores(retInfo);
        }

        try {
            if (numVariants > 0)
                knnQueries = knnRelModel.getKNNs(q, numVariants);

            if (knnQueries!=null && !knnQueries.isEmpty()) {
                variantSpec = variantSpecificity(q, knnQueries, retInfo, topDocs, k, verbose);
            }

        }
        catch (Exception ex) { ex.printStackTrace(); }

        if(verbose & (knnQueries!=null)) {
            System.out.printf("target=%f, variant=%f \n", baseModel.computeSpecificity(q, retInfo, topDocs, k), variantSpec);
        }

        return knnQueries!=null?
                lambda * variantSpec + (1-lambda) * baseModel.computeSpecificity(q, retInfo, topDocs, k) / this.scaler:
                baseModel.computeSpecificity(q, retInfo, topDocs, k);
    }

    double variantSpecificity(MsMarcoQuery q, List<MsMarcoQuery> knnQueries,
                              RetrievedResults retInfo, TopDocs topDocs, int k, boolean verbose) throws Exception {
        double specScore = 0;
        double z = 0;
        double variantSpecScore;
        double refSim;

        // apply QPP base model on these estimated relevance scores
        // System.out.print("rqs used here are: ");
        for (MsMarcoQuery rq: knnQueries) {
            //System.out.println(rq.toString());
            // System.out.print(rq.getId());
            // System.out.print(" ");
            TopDocs topDocsRQ = searcher.search(rq.getQuery(), k);

            RetrievedResults varInfo;
            if(qvResults == null) {
                varInfo = new RetrievedResults(rq.getId(), topDocsRQ);
            } else {
                varInfo = qvResults.getRetrievedResultsForQueryId(rq.getId());
                if(varInfo == null) {
                    System.out.printf("%s NO RECORD!!\n", rq.getId());
                    continue;
                }
            }

            variantSpecScore = baseModel.computeSpecificity(rq, varInfo, topDocs, k, verbose);

            //if nothing has been retrieved, then set the weight to 0
            if(variantSpecScore == -1){
                refSim = 0;
            } else {
                refSim = rq.getRefSim();
            }

            //System.out.println(String.format("%s %.4f", rq.getId(), variantSpecScore));
            specScore +=  refSim * variantSpecScore ;
            z += refSim;

            if(verbose) {
                System.out.println(String.format("rqid=%s, sim=%f, est=%f", rq.getId(), refSim, variantSpecScore));
            }
        }
        
        return z==0? baseModel.computeSpecificity(q, retInfo, topDocs, k, verbose): specScore/z;
    }

}
