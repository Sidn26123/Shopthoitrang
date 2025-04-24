package ptithcm.common;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class timeHelper {

    // Hàm format thời gian
    public static String formatDatetime(String timeString) {
        // Định dạng chuỗi thời gian đầu vào
        DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
        LocalDateTime dateTime = LocalDateTime.parse(timeString, inputFormatter);
        System.out.println("A");
        // Định dạng chuỗi đầu ra theo yêu cầu hh:mm:ss dd/MM/yyyy
        DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern("HH:mm:ss dd/MM/yyyy");
        return dateTime.format(outputFormatter);
    }

    public static String formatDatetime(String timeString, String inputFormat, String outputFormat) {
        // Định dạng chuỗi thời gian đầu vào
        DateTimeFormatter inputFormatter = DateTimeFormatter.ofPattern(inputFormat);
        LocalDateTime dateTime = LocalDateTime.parse(timeString, inputFormatter);

        // Định dạng chuỗi đầu ra theo yêu cầu hh:mm:ss dd/MM/yyyy
        DateTimeFormatter outputFormatter = DateTimeFormatter.ofPattern(outputFormat);
        return dateTime.format(outputFormatter);
    }
}
