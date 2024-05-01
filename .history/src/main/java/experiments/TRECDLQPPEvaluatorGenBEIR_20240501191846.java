package experiments;

import correlation.KendalCorrelation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import qpp.NQCSpecificity;
import qpp.QPPMethod;
import qpp.UEFSpecificity;
import qpp.VariantSpecificity;
import qpp.CoRelSpecificity;
import qrels.Evaluator;
import qrels.Metric;
import qrels.RetrievedResults;
import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;
import qrels.AllRetrievedResults; // added 0421

import java.util.List;
import java.util.Map;

public class TRECDLQPPEvaluatorGenBEIR {

    static String QUERY_FILE_MSMARCO = "data/trecdl/pass_full.queries";
    static String QRELS_FILE_MSMARCO = "data/trecdl/pass_full.qrels";
    
    static double scaler = -1;

    static void updateScaler(
        Evaluator evaluator,
        List<MsMarcoQuery> queries) {
        
        double scalerR = 0;
        int countScaler = 0;
        for (MsMarcoQuery query : queries) {

            RetrievedResults rr = evaluator.getRetrievedResultsForQueryId(query.getId());

            double[] scoreList = rr.getRSVs(50);

            scalerR += calculateVariation(scoreList);
            countScaler ++;
        }

        scaler = (scalerR/countScaler) * 1;
        System.out.println(scaler);
    }

    static double runExperiment(
            String baseQPPModelName, // nqc/uef
            IndexSearcher searcher,
            KNNRelModel knnRelModel,
            Evaluator evaluator,
            List<MsMarcoQuery> queries,
            Map<String, TopDocs> topDocsMap,
            float lambda, int numVariants, Metric targetMetric,
            AllRetrievedResults qvResults) {

        double kendals = 0;

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
        kendals = new KendalCorrelation().correlation(evaluatedMetricValues, qppEstimates);
        return kendals;
    }

    static double trainAndTest(
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
            String variantsFile,
            String variantsQidFile,
            String scoreFile,
            boolean extendToRelQueryFromDocs,
            boolean useRBO,
            AllRetrievedResults qvResults // for testing same retriever
    )
            throws Exception {
        Settings.init(retrieverTrain.getSearcher());
        IndexSearcher searcher = retriever.getSearcher();
        KNNRelModel knnRelModel;
        if(!scoreFile.equals("")){
            knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, trainQueryFile, variantsFile, variantsQidFile, scoreFile, extendToRelQueryFromDocs, useRBO);
        } else {
            knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, trainQueryFile, variantsFile, useRBO);
        }
        List<MsMarcoQuery> trainQueries = knnRelModel.getQueries();

        Evaluator evaluatorTrain = new Evaluator(trainQrelsFile, trainResFile); // load ret and rel
        Map<String, TopDocs> topDocsMap = evaluatorTrain.getAllRetrievedResults().castToTopDocs();

        OptimalHyperParams p = new OptimalHyperParams();

        for (int numVariants=1; numVariants<=maxNumVariants; numVariants++) {
            for (float l = 0; l <= 1.0; l += Constants.QPP_COREL_LAMBDA_STEPS) {
                double kendals = runExperiment(baseModelName,
                        searcher, knnRelModel, evaluatorTrain,
                        trainQueries, topDocsMap, l, numVariants, targetMetric,
                        qvResults);

                System.out.println(String.format("Train on %s -- (%.1f, %d): tau = %.4f",
                        trainQueryFile, l, numVariants, kendals));
                if (kendals > p.kendals) {
                    p.l = l;
                    p.numVariants = numVariants;
                    p.kendals = kendals; // keep track of max
                }
            }
        }
        System.out.println(String.format("The best settings: lambda=%.1f, nv=%d", p.l, p.numVariants));
        // apply this setting on the test set
        KNNRelModel knnRelModelTest;
            
        if(!scoreFile.equals("")){
            knnRelModelTest = new KNNRelModel(Constants.QRELS_TRAIN, testQueryFile, variantsFile, variantsQidFile, scoreFile, extendToRelQueryFromDocs, useRBO);
        } else {
            knnRelModelTest = new KNNRelModel(Constants.QRELS_TRAIN, testQueryFile, variantsFile, useRBO);
        }
        
        List<MsMarcoQuery> testQueries = knnRelModelTest.getQueries(); // these queries are different from train queries

        Evaluator evaluatorTest = new Evaluator(testQrelsFile, testResFile); // load ret and rel

        Map<String, TopDocs> topDocsMapTest = evaluatorTest.getAllRetrievedResults().castToTopDocs();
        double kendals_Test = runExperiment(baseModelName,
                searcher, knnRelModelTest,
                evaluatorTest, testQueries, topDocsMapTest, p.l, p.numVariants, targetMetric,
                qvResults);

        System.out.println(String.format(
                "Kendal's on %s with lambda=%.1f, M=%d: %.4f",
                testQueryFile, p.l, p.numVariants, kendals_Test));

        return kendals_Test;
    }

    static void runSingleExperiment(
            String baseModelName,
            OneStepRetriever retriever,
            String queryFile, String qrelsFile,
            String resFile,
            Metric targetMetric,
            int numVariants,
            float l,
            String variantFile,
            boolean useRBO,
            AllRetrievedResults qvResults
    )
            throws Exception {

        KNNRelModel knnRelModel = new KNNRelModel(Constants.QRELS_TRAIN, queryFile, variantFile, useRBO);
        List<MsMarcoQuery> testQueries = knnRelModel.getQueries(); // these queries are different from train queries

        Evaluator evaluatorTest = new Evaluator(qrelsFile, resFile); // load ret and rel

        Map<String, TopDocs> topDocsMapTest = evaluatorTest.getAllRetrievedResults().castToTopDocs();
        double kendals = runExperiment(baseModelName, retriever.getSearcher(),
                knnRelModel, evaluatorTest, testQueries, topDocsMapTest,
                l, numVariants, targetMetric, qvResults);
        System.out.println(String.format("Target Metric: %s, tau = %.4f", targetMetric.toString(), kendals));
    }

    public static double calculateVariation(double[] array) {

        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }
    
        // get the mean of array
        int length = array.length;
        double mean = sum / length;
    
        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }
    
        return standardDeviation / length;
    }

    public static void main(String[] args) {

        if (args.length < 7) {
            System.out.println("Required arguments: <res file DL 19> <res file DL 20> <metric (ap/ndcg)> <uef/nqc> <rlm/w2v (variant gen)> <extend queries(1)?>");
            args = new String[7];
            args[0] = "runs/splade.dl19.100.pp";
            args[1] = "runs/splade.dl20.100.pp";
            args[2] = "ap";
            args[3] = "nqc";
            args[4] = "rlm";
            args[5] = "false"; //extend to doc->query
            args[6] = "false"; //useRBO
        }

        String resFileForMSMARCO = args[0];
        String resFileForBEIR = args[1];
        Metric targetMetric = args[2].equals("ap")? Metric.AP : Metric.nDCG;
        // String variantFile = args[4].equals("rlm")? Constants.QPP_JM_VARIANTS_FILE_RLM: Constants.QPP_JM_VARIANTS_FILE_W2V;
        String variantFile = "";
        String variantQidFile = "";
        String scoreFile = "";
        String rqResFile = "";

        String retrieverName = "bm25";

        if (args[0].indexOf("mt5") != -1) {
            // mt5
            retrieverName = "mt5";
        } else if (args[0].indexOf("BM25+BERT") != -1){
            // bert
            retrieverName = "bert";
        }

        // if(Constants.SAME_RETRIEVER) { // change the rqResFile according to retriever+qv_type
        //     if(retrieverName.equals("mt5")){
        //         rqResFile = String.format("%s/QV_bm25_%s_%s.res", Constants.QV_RESFILE_BASE_PATH, retrieverName, args[4]);
        //     }
        // }

        //here ..... 0428
        String corpusIndex = Constants.COVID_INDEX;
        String QUERY_FILE_BEIR = Constants.COVID_QUERIES;
        String QRELS_FILE_BEIR = Constants.COVID_QRELS;
        if(resFileForBEIR.indexOf("fever") != -1){
            corpusIndex = Constants.FEVER_INDEX;
            QUERY_FILE_BEIR = Constants.FEVER_QUERIES;
            QRELS_FILE_BEIR = Constants.FEVER_QRELS;
        } else if (resFileForBEIR.indexOf("touche") != -1){
            corpusIndex = Constants.WEBIS_INDEX;
            QUERY_FILE_BEIR = Constants.WEBIS_QUERIES;
            QRELS_FILE_BEIR = Constants.WEBIS_QRELS;
        }

        //here ..... 0421
        AllRetrievedResults qvResults = null;
        if( ! rqResFile.equals("")) {
            qvResults = new AllRetrievedResults(rqResFile); //rqResFile should be a constant
            // System.out.println(qvResults);
        }

        switch(args[4]){ 
            case "rlm":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_RLM;
                break;
            case "w2v":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_W2V;
                break;
            default:
                variantFile = Constants.QPP_JM_VARIANTS_FILE_SBERT;
                variantQidFile = Constants.QPP_JM_VARIANTS_QID_FILE_SBERT;
                scoreFile = Constants.QPP_JM_SCORE_FILE_SBERT;
        }
        
        boolean extendOne = Boolean.parseBoolean(args[5]);
        boolean useRBO = Boolean.parseBoolean(args[6]);

        try {
            OneStepRetriever retrieverMSMARCO = new OneStepRetriever(Constants.QUERY_FILE_TEST, Constants.MSMARCO_INDEX);
            OneStepRetriever retrieverBEIR = new OneStepRetriever(Constants.QUERY_FILE_TEST, corpusIndex);

            /*
            for (int i=0; i<=1; i++) {
                runSingleExperiment(args[3], retriever, QUERY_FILES[i], QRELS_FILES[i], args[i], targetMetric, 3, 0.5f, variantFile);
            }
            System.exit(0);
            */

            // read QV res file if the filename is not == ""


            double kendalsOnTest = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL19], QRELS_FILES[DL19],
                    QUERY_FILES[DL20], QRELS_FILES[DL20],
                    args[0], args[1], Constants.QPP_COREL_MAX_VARIANTS, 
                    variantFile, variantQidFile, scoreFile, 
                    extendOne, useRBO, qvResults);
            double kendalsOnTrain = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL20], QRELS_FILES[DL20],
                    QUERY_FILES[DL19], QRELS_FILES[DL19],
                    args[1], args[0], Constants.QPP_COREL_MAX_VARIANTS, 
                    variantFile, variantQidFile, scoreFile, 
                    extendOne, useRBO, qvResults);

            double kendals = 0.5*(kendalsOnTrain + kendalsOnTest);
            System.out.println(String.format("Target Metric: %s, tau = %.4f", targetMetric.toString(), kendals));
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

}
