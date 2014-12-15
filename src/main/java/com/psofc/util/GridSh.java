package com.psofc.util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * The class is for output a script file for a data set
 * This class is independently depend on the directory of the dataset.
 * @author YAN
 *
 */
public class GridSh {

	static String type = "multi_array";
	static String trCls = "multi_final_10folds";

	public static void main(String[] args) throws IOException {

		String dir = "shs_" + type + "/";

		GridSh gs = new GridSh(dir);

		String[] files = OrgClassification.getAllDataName();

		if(files == null){System.err.println("No files read");return;}


		for(String fn:files){
			gs.output(fn, gs.setContent(fn, "run_array_mul_oct_8_10folds"));
		}


	}


	String dir;

	public GridSh(String dir){this.dir = dir;}



	public String setContent(String fname, String target_folder){

		String content = "#$ -S /bin/sh \n"
				+ "#$ -wd /vol/grid-solar/sgeusers/daiyan\n"
				+ "##$ -M Yan.Dai@ecs.vuw.ac.nz\n" + "##$ -m be \n" + "\n" +

				"mkdir -p /local/tmp/daiyan/$JOB_ID \n\n"
				+ "if [ -d /local/tmp/daiyan/$JOB_ID ]; then\n"
				+ "        cd /local/tmp/daiyan/$JOB_ID\n" + "else\n"
				+ "        echo \"There's no job directory to change into\" \n"
				+ "        echo \"Here's LOCAL TMP \"\n"
				+ "        ls -la /local/tmp\n"
				+ "        echo \"AND LOCAL TMP FRED \"\n"
				+ "        ls -la /local/tmp/daiyan\n"
				+ "        echo \"Exiting\"\n" + "        exit 1\n\n" +

				"fi\n\n" +

				"cp /vol/grid-solar/sgeusers/daiyan/" + target_folder
				+ "/" + type + "java.jar .\n" + "mkdir file_" + type + "\n"
				+ "cp -r /vol/grid-solar/sgeusers/daiyan/Data .\n" +

				"java -jar " + type + "java.jar " + fname + "\n\n" +

				"rm "+ type + "java.jar\n"
				+ "cp -r *  /vol/grid-solar/sgeusers/daiyan/" + target_folder
				+ "/\n" + "rm -r Data\n" + "rm -r file_" + type + "\n"
				+ "cd /vol/grid-solar/sgeusers/daiyan\n" + "pwd\n"
				+ "rm -fr /local/tmp/daiyan/$JOB_ID\n";

		return content;
	}


	void output(String fname, String content) throws IOException{
		String file = this.dir + fname +"_" + type + "_" + trCls + "_script.sh";

		PrintWriter wf = new PrintWriter(new FileWriter(file));

		wf.append(content);

		wf.close();


	}




}
