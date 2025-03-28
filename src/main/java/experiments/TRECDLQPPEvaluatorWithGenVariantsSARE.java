package experiments;

import correlation.KendalCorrelation;
import correlation.SARE;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TopDocs;
import qpp.NQCSpecificity;
import qpp.QPPMethod;
import qpp.UEFSpecificity;
import qpp.VariantSpecificity;
import qrels.AllRetrievedResults;
import qrels.Evaluator;
import qrels.Metric;
import qrels.RetrievedResults;
import retrieval.Constants;
import retrieval.KNNRelModel;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;

import java.util.List;
import java.util.Map;

public class TRECDLQPPEvaluatorWithGenVariantsSARE {

    static final int DL19 = 0;
    static final int DL20 = 1;
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

        double tau = 0;
        double sare = 0;

        QPPMethod baseModel = baseQPPModelName.equals("nqc")? new NQCSpecificity(searcher): new UEFSpecificity(new NQCSpecificity(searcher));

        VariantSpecificity qppMethod = new VariantSpecificity(
                    baseModel,
                    searcher,
                    knnRelModel,
                    numVariants,
                    lambda
        );
        qppMethod.setQvResults(qvResults);

        int numQueries = queries.size();
        double[] qppEstimates = new double[numQueries];
        double[] evaluatedMetricValues = new double[numQueries];
        String[] qids = new String[numQueries];

        int i = 0;       

        for (MsMarcoQuery query : queries) {

            RetrievedResults rr = evaluator.getRetrievedResultsForQueryId(query.getId());

            TopDocs topDocs = topDocsMap.get(query.getId());

            evaluatedMetricValues[i] = evaluator.compute(query.getId(), targetMetric);
            qppEstimates[i] = (float) qppMethod.computeSpecificity(
                    query, rr, topDocs, Constants.QPP_NUM_TOPK, false);
            qids[i] = query.getId();

            i++;
        }

        tau = new KendalCorrelation().correlation(evaluatedMetricValues, qppEstimates);
        sare = new SARE().correlationWithLog(evaluatedMetricValues, qppEstimates, qids);

        return new TauAndSARE(tau, sare);
    }

    static TauAndSARE trainAndTest(
            String baseModelName,
            OneStepRetriever retriever,
            Metric targetMetric,
            String trainQueryFile,
            String trainQrelsFile,
            String trainResFile,
            String variantsFile,
            String variantsQidFile,
            String scoreFile,
            boolean extendToRelQueryFromDocs,
            boolean useRBO,
            AllRetrievedResults qvResults, // for testing same retriever
            int numVariants,
            float l

    )
            throws Exception {
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

//
//        int numVariants = 5;
//        float l = (float)0.5;

        TauAndSARE analyseResults = runExperiment(baseModelName,
                        searcher, knnRelModel, evaluatorTrain,
                        trainQueries, topDocsMap, l, numVariants, targetMetric,
                        qvResults);

        return analyseResults;
    }

    public static void main(String[] args) {

        if (args.length < 8) {
            System.out.println("Required arguments: <res file DL 19> <res file DL 20> <metric (ap/ndcg)> <uef/nqc> <rlm/w2v (variant gen)> <extend queries(1)?> <k> <lambda>");
            args = new String[8];
            args[0] = "runs/splade.dl19.100.pp";
            args[1] = "runs/splade.dl20.100.pp";
            args[2] = "ap";
            args[3] = "nqc";
            args[4] = "sbert"; //qv method (to retrieve the k-shot example)
            args[5] = "false"; //extend to doc->query
//            args[6] = "1"; //number of shots
            args[6] = "1";
            args[7] = "0.9";
        }

        Metric targetMetric = args[2].equals("ap")? Metric.AP : Metric.nDCG;
        String variantFile = "";
        String variantQidFile = "";
        String scoreFile = "";

        //here ..... 0421
        AllRetrievedResults qvResults = null;

        switch(args[4]){
            case "rlm":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_RLM;
                break;
            case "w2v":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_W2V;
                break;
            case "gpt":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_GPT;
                break;
            case "llama3":
                variantFile = Constants.QPP_JM_VARIANTS_FILE_LLAMA3;
                break;
            default:
                variantFile = Constants.QPP_JM_VARIANTS_FILE_SBERT;
                variantQidFile = Constants.QPP_JM_VARIANTS_QID_FILE_SBERT;
                scoreFile = Constants.QPP_JM_SCORE_FILE_SBERT;
        }
        
        boolean extendOne = Boolean.parseBoolean(args[5]);
        boolean useRBO = true;
        try {
            OneStepRetriever retriever = new OneStepRetriever(Constants.QUERY_FILE_TEST);
            Settings.init(retriever.getSearcher());

            TauAndSARE sareOn19 = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL19], QRELS_FILES[DL19], args[0],
                    variantFile, variantQidFile, scoreFile, 
                    extendOne, useRBO, qvResults, Integer.parseInt(args[6]), Float.parseFloat(args[7]));

            System.out.println(String.format("TREC-19: Target Metric: %s, tau = %.4f, sARE = %.4f",
                     targetMetric.toString(), sareOn19.tau, sareOn19.sare));

            TauAndSARE sareOn20 = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL20], QRELS_FILES[DL20], args[1],
                    variantFile, variantQidFile, scoreFile, 
                    extendOne, useRBO, qvResults, Integer.parseInt(args[6]), Float.parseFloat(args[7]));

            System.out.println(String.format("TREC-20: Target Metric: %s, tau = %.4f, sARE = %.4f",
                    targetMetric.toString(), sareOn20.tau, sareOn20.sare));

            TauAndSARE sareOn19Base = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL19], QRELS_FILES[DL19], args[0],
                    variantFile, variantQidFile, scoreFile,
                    extendOne, useRBO, qvResults, Integer.parseInt(args[6]), (float)0);

            System.out.println(String.format("Base-TREC-19: Target Metric: %s, tau = %.4f, sARE = %.4f",
                    targetMetric.toString(), sareOn19Base.tau, sareOn19Base.sare));

            TauAndSARE sareOn20Base = trainAndTest(args[3], retriever, targetMetric,
                    QUERY_FILES[DL20], QRELS_FILES[DL20], args[1],
                    variantFile, variantQidFile, scoreFile,
                    extendOne, useRBO, qvResults, Integer.parseInt(args[6]), (float)0);

            System.out.println(String.format("Base-TREC-20: Target Metric: %s, tau = %.4f, sARE = %.4f",
                    targetMetric.toString(), sareOn20Base.tau, sareOn20Base.sare));
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
