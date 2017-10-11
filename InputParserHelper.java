import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Scanner;
import java.lang.StringBuilder;

public class InputParserHelper {
	int fileSizeLimit = 1500000;
	public static void main(String[] args) {
		new InputParserHelper();
	}

	public InputParserHelper() {
		BufferedReader in = null;
		String[] input = new String[fileSizeLimit];

		try {
		    File file = new File("training.1600000.processed.noemoticon.csv");
		    in = new BufferedReader(new FileReader(file));

		    String line;
		    int i = 0;
		    while ((line = in.readLine()) != null && i < fileSizeLimit) {
		        input[i] = line;
		        i++;
		    }

		} catch (IOException e) {
		    e.printStackTrace();
		} finally {
		    try {
		        in.close();
		    } catch (IOException e) {
		        e.printStackTrace();
		    }
		}

		parseInput(input);
	}

	public void parseInput(String[] input) {
		Scanner sc = new Scanner(System.in);
		String[] row = new String[4];
		String stringToFile[][] = new String[fileSizeLimit][2];
		int pos = 0;
		

		for(String line : input) {
			char[] l = line.toCharArray();
			int commaCounter = 0;
			boolean inTimestamp = false;
			StringBuilder date = new StringBuilder();
			StringBuilder tweet = new StringBuilder();

			for(char c : l) {
				if(c == ',') {
					commaCounter++;
				} else{
					if(commaCounter == 2) {
						if(c != '"') {
							date.append(c);
						}
					} else if(commaCounter == 5) {
						if(c != '"') {
							tweet.append(c);
						}
					}
				}
			}
			stringToFile[pos][0] = date.toString();
			stringToFile[pos][1] = tweet.toString();
			pos++;
		}
		try {
			PrintWriter mon = new PrintWriter("MonTwitterSentiment.txt", "UTF-8");
			PrintWriter tue = new PrintWriter("TueTwitterSentiment.txt", "UTF-8");
			PrintWriter wed = new PrintWriter("WedTwitterSentiment.txt", "UTF-8");
			PrintWriter thu = new PrintWriter("ThuTwitterSentiment.txt", "UTF-8");
			PrintWriter fri = new PrintWriter("FriTwitterSentiment.txt", "UTF-8");
			PrintWriter sat = new PrintWriter("SatTwitterSentiment.txt", "UTF-8");
			PrintWriter sun = new PrintWriter("SunTwitterSentiment.txt", "UTF-8");

			for(int i = 0; i < fileSizeLimit; i++) {
				if(stringToFile[i][0].substring(0,3).toLowerCase().equals("mon")) {
					mon.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("tue")) {
					tue.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("wed")) {
					wed.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("thu")) {
					thu.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("fri")) {
					fri.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("sat")) {
					sat.println(stringToFile[i][1]);
				} else if(stringToFile[i][0].substring(0,3).toLowerCase().equals("sun")) {
					sun.println(stringToFile[i][1]);
				}

			}
			
			mon.close();
			tue.close();
			wed.close();
			thu.close();
			fri.close();
			sat.close();
			sun.close();
		}catch (IOException e) {
		    // do something
			
		}
		

	}
}