package experiments;

import correlation.SARE;

import org.apache.commons.math3.analysis.function.Constant;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import correlation.KendalCorrelation;
import qrels.*;
import qpp.*;

import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class OptimalHyperParams {
    float l;
    float m;
    int numNeighbors;
    int numVariants;
    double kendals;
}

public class TRECDLQPPEvaluatorBEIR {
    // static final int DL19 = 0;
    // static final int DL20 = 1;
    static String[] QUERY_FILES = {"data/trecdl/pass_2019.queries", "data/trecdl/pass_2020.queries"};
    static String[] QRELS_FILES = {"data/trecdl/pass_2019.qrels", "data/trecdl/pass_2020.qrels"};

    static class TauAndSARE {
        double tau;
        double sare;

        TauAndSARE(double tau, double sare) {
            this.tau = tau;
            this.sare = sare;
        }
    }

    static TauAndSARE runExperiment(
            String baseQPPModelName, // nqc/uef
            IndexSearcher searcher,
            KNNRelModel knnRelModel,
            Evaluator evaluator,
            List<MsMarcoQuery> queries,
            Map<String, TopDocs> topDocsMap,
            float lambda, int numVariants, Metric targetMetric,
            AllRetrievedResults qvResults) {

        QPPMethod baseModel = baseQPPModelName.equals("nqc")? new NQCSpecificity(searcher): new UEFSpecificity(new NQCSpecificity(searcher));

        boolean useClarity = Constants.USE_CLARITY; // hard coded temporarily
        VariantSpecificity qppMethod;
        if(useClarity){
            qppMethod = new CoRelSpecificity(
                    baseModel,
                    searcher,
                    knnRelModel,
                    numVariants,
                    lambda
            );             
        } else {
            qppMethod = new VariantSpecificity(
                    baseModel,
                    searcher,
                    knnRelModel,
                    numVariants,
                    lambda
            ); // I changed it to the subclass, is it ok?
            // qppMethod.setScaler(scaler);
            qppMethod.setQvResults(qvResults); // for M=M' test
        } 

        int numQueries = queries.size();
        double[] qppEstimates = new double[numQueries];
        double[] evaluatedMetricValues = new double[numQueries];

        int i = 0;

        for (MsMarcoQuery query : queries) {
            RetrievedResults rr = evaluator.getRetrievedResultsForQueryId(query.getId());

            TopDocs topDocs = topDocsMap.get(query.getId());

            evaluatedMetricValues[i] = evaluator.compute(query.getId(), targetMetric);
            qppEstimates[i] = (float) qppMethod.computeSpecificity(
                    query, rr, topDocs, Constants.QPP_NUM_TOPK);

            //System.out.println(String.format("%s: AP = %.4f, QPP = %.4f", query.getId(), evaluatedMetricValues[i], qppEstimates[i]));
            i++;
        }
        //System.out.println(String.format("Avg. %s: %.4f", targetMetric.toString(), Arrays.stream(evaluatedMetricValues).sum()/(double)numQueries));

        double tau = new KendalCorrelation().correlation(evaluatedMetricValues, qppEstimates);
        double sare = new SARE().correlation(evaluatedMetricValues, qppEstimates);

        return new TauAndSARE(tau, sare);
    }

    static TauAndSARE trainAndTest(
            String baseModelName,
            OneStepRetriever retriever,
            Metric targetMetric,
            String trainQueryFile,
            String trainQrelsFile,
            String testQueryFile,
            String testQrelsFile,
            String trainResFile,
            String testResFile,
            int maxNumVariants,
            boolean useRBO,
            boolean extendQV,
            AllRetrievedResults qvResults // for testing same retriever
    )
    throws Exception {
        IndexSearcher searcher = retriever.getSearcher();
        KNNRelModel knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, trainQueryFile, useRBO, extendQV);
        List<MsMarcoQuery> trainQueries = knnRelModel.getQueries();

        Evaluator evaluatorTrain = new Evaluator(trainQrelsFile, trainResFile); // load ret and rel
        QPPEvaluator qppEvaluator = new QPPEvaluator(
                trainQueryFile, trainQrelsFile,
                new KendalCorrelation(), retriever.getSearcher(), Constants.QPP_NUM_TOPK);

        Map<String, TopDocs> topDocsMap = evaluatorTrain.getAllRetrievedResults().castToTopDocs();

        OptimalHyperParams p = new OptimalHyperParams();

        for (int numVariants=1; numVariants<=maxNumVariants; numVariants++) {
            for (float l = 0; l <= 1.0; l += Constants.QPP_COREL_LAMBDA_STEPS) {
                TauAndSARE tauAndSARE = runExperiment(baseModelName,
                        searcher, knnRelModel, evaluatorTrain,
                        trainQueries, topDocsMap, 
                        l, numVariants, targetMetric,
                        qvResults);

                System.out.println(String.format("Train on %s -- (%.1f, %d): tau = %.4f",
                        trainQueryFile, l, numVariants, tauAndSARE.tau, tauAndSARE.sare));
                if (tauAndSARE.tau > p.kendals) {
                    p.l = l;
                    p.numVariants = numVariants;
                    p.kendals = tauAndSARE.tau; // keep track of max
                }
            }
        }
        System.out.println(String.format("The best settings: lambda=%.1f, nv=%d", p.l, p.numVariants));
        // apply this setting on the test set
        KNNRelModel knnRelModelTest = new KNNRelModel(Constants.QRELS_TRAIN, testQueryFile, useRBO, extendQV);
        List<MsMarcoQuery> testQueries = knnRelModelTest.getQueries(); // these queries are different from train queries

        Evaluator evaluatorTest = new Evaluator(testQrelsFile, testResFile); // load ret and rel
        QPPEvaluator qppEvaluatorTest = new QPPEvaluator(
                testQueryFile, testQrelsFile,
                new KendalCorrelation(), retriever.getSearcher(), Constants.QPP_NUM_TOPK);

        Map<String, TopDocs> topDocsMapTest = evaluatorTest.getAllRetrievedResults().castToTopDocs();
        TauAndSARE tauAndSARE_Test = runExperiment(baseModelName,
                searcher, knnRelModelTest,
                evaluatorTest, testQueries, topDocsMapTest, 
                p.l, p.numVariants, targetMetric,
                qvResults);

        System.out.println(String.format(
                "Kendal's on %s with lambda=%.1f, M=%d: %.4f %.4f",
                testQueryFile, p.l, p.numVariants, tauAndSARE_Test.tau, tauAndSARE_Test.sare));

        return tauAndSARE_Test;
    }

    static void runSingleExperiment(
            String baseModelName,
            OneStepRetriever retriever,
            String queryFile, String qrelsFile,
            String resFile,
            Metric targetMetric,
            int numVariants,
            float l,
            boolean useRBO,
            boolean extendQV,
            AllRetrievedResults qvResults
    )
    throws Exception {

        KNNRelModel knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, queryFile, useRBO, extendQV);
        Evaluator evaluatorTest = new Evaluator(qrelsFile, resFile); // load ret and rel
        QPPEvaluator qppEvaluatorTest = new QPPEvaluator(
                queryFile, qrelsFile,
                new KendalCorrelation(), retriever.getSearcher(), Constants.QPP_NUM_TOPK);
        List<MsMarcoQuery> testQueries = qppEvaluatorTest.constructQueries(queryFile); // these queries are different from train queries

        Map<String, TopDocs> topDocsMapTest = evaluatorTest.getAllRetrievedResults().castToTopDocs();
        TauAndSARE tauAndSARE = runExperiment(baseModelName, retriever.getSearcher(),
                                        knnRelModel, evaluatorTest, testQueries, topDocsMapTest,
                                        l, numVariants, targetMetric, qvResults);
        System.out.println(String.format("Target Metric: %s, tau = %.4f sARE = %.4f", targetMetric.toString(), tauAndSARE.tau, tauAndSARE.sare));
    }

    public static void main(String[] args) {

        if (args.length < 4) {
            System.out.println("Required arguments: <res file for training (MSMarco)> <res file for testing (BEIR)> <metric (ap/ndcg)> <uef/nqc> <corpus>");
            args = new String[5];
            args[0] = "runs/train.covid.mt5.res"; // from msmarco queries
            args[1] = "runs/test.covid.mt5.res"; // from beir queries
            args[2] = "ap";
            args[3] = "nqc";
            args[4] = "true";
            args[5] = "false";
        }

        Metric targetMetric = args[2].equals("ap")? Metric.AP : Metric.nDCG;
        boolean useRBO = Boolean.parseBoolean(args[4]);
        boolean extendQV = Boolean.parseBoolean(args[5]);
        
        String rqResFile = "";
        String retrieverName = "bm25";

        if (args[0].indexOf("mt5") != -1) {
            // mt5
            retrieverName = "mt5";
        } else if (args[0].indexOf("BM25+BERT") != -1){
            // bert
            retrieverName = "bert";
        }

        if(Constants.SAME_RETRIEVER) { // change the rqResFile according to retriever+qv_type
            if(retrieverName.equals("mt5")){
                rqResFile = String.format("%s/QV_bm25_%s_%s.res", Constants.QV_RESFILE_BASE_PATH, retrieverName, "bm25");
            }
        }

        //here ..... 0428
        String corpusIndex = Constants.COVID_INDEX;
        if(args[0].indexOf("fever") != -1){
            corpusIndex = Constants.FEVER_INDEX;
        } else if (args[0].indexOf("touche") != -1){
            corpusIndex = Constants.WEBIS_INDEX;
        }

        //here ..... 0421
        AllRetrievedResults qvResults = null;
        if( ! rqResFile.equals("")) {
            qvResults = new AllRetrievedResults(rqResFile); //rqResFile should be a constant
            // System.out.println(qvResults);
        }

        try {
            OneStepRetriever retriever = new OneStepRetriever(Constants.QUERY_FILE_TEST, corpusIndex);
            Settings.init(retriever.getSearcher());

            TauAndSARE kendalsOnTest = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL19], QRELS_FILES[DL19],
                    QUERY_FILES[DL20], QRELS_FILES[DL20],
                    args[0], args[1], Constants.QPP_COREL_MAX_VARIANTS, 
                    useRBO, extendQV,
                    qvResults);
            TauAndSARE kendalsOnTrain = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL20], QRELS_FILES[DL20],
                    QUERY_FILES[DL19], QRELS_FILES[DL19],
                    args[1], args[0], Constants.QPP_COREL_MAX_VARIANTS, 
                    useRBO, extendQV,
                    qvResults);

            double kendals = 0.5*(kendalsOnTrain.tau + kendalsOnTest.tau);
            double sare = 0.5*(kendalsOnTrain.sare + kendalsOnTest.sare);
            System.out.println(String.format("Target Metric: %s, tau = %.4f, sare = %.4f", targetMetric.toString(), kendals, sare));
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}
