package ptithcm.controller;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import ptithcm.entity.SanPhamEntity;
import ptithcm.service.SanPhamService;

@Transactional
@Controller
@RequestMapping()
public class imageSearchController {
	
	@Autowired
	SanPhamService sanPhamService;
	
	@RequestMapping("/imageSearch")
	public String imageSearch(HttpServletRequest request, ModelMap model) {
		return "deXuat/imageSearch";
	}
	
//	String filePath = "D:\TAI LIEU KY 6\LAP TRINH WEB\WEB BAO CAO\WEB BAO CAO\GITHUB\FashionSalesWebsite\\ShopThoiTrang\\src\\main\\webapp\\assets\\img\\imgSearch\\";
	String filePath = "D:\\TAI LIEU KY 6\\LAP TRINH WEB\\imgSearch\\";
	String pythonPath = "D:\\TAI LIEU KY 6\\LAP TRINH WEB\\WEB BAO CAO\\WEB BAO CAO\\GITHUB\\FashionSalesWebsite\\ShopThoiTrang\\src\\main\\python";
	@RequestMapping(value="/imageSearch", params = "btnSearchImg", method = RequestMethod.POST)
	public String imageSearch(HttpServletRequest request, @RequestParam("avatar") MultipartFile avatar, ModelMap model) throws IOException {
		
		String avatarFileName = avatar.getOriginalFilename();
	    String avatarFilePath = filePath + avatarFileName;
	    File avatarFile = new File(avatarFilePath);
	    avatar.transferTo(avatarFile);
		
		  String param = avatarFilePath;
		  ProcessBuilder builder = new ProcessBuilder(
	                "cmd.exe", "/c", 
	                "cd \"D:\\\\TAI LIEU KY 6\\\\LAP TRINH WEB\\\\WEB BAO CAO\\\\WEB BAO CAO\\\\GITHUB\\\\FashionSalesWebsite\\ShopThoiTrang\\src\\main\\python\" && " +
	                "python \"D:\\\\TAI LIEU KY 6\\\\LAP TRINH WEB\\\\WEB BAO CAO\\\\WEB BAO CAO\\\\GITHUB\\\\FashionSalesWebsite\\ShopThoiTrang\\src\\main\\python\\search_image.py\" \"" + param + "\""
	            );
	        builder.redirectErrorStream(true);
	        Process p = builder.start();
	        BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
	        	System.out.println("test anh");
	         
	     // Đọc đầu ra từ quy trình Python và lưu vào danh sách
	        List<String> imageNames = new ArrayList<>();
	        String line;
	        while ((line = r.readLine()) != null) {
	        	System.out.println(line);
	            // Kiểm tra xem dòng có chứa tên tệp ảnh không
	            if (line.startsWith("name: ")) {
	            	String imageName = line.replace("name: ", "");
	                imageNames.add(imageName);
	            }
	        }
	        
	        if (imageNames.isEmpty()) {
	            // Nếu danh sách imageNames rỗng, truyền thông báo vào model
	            model.addAttribute("message", "Không có sản phẩm cần tìm.");
	        } else {
	            // Nếu danh sách không rỗng, tiến hành tìm sản phẩm và truyền vào model
	            List<SanPhamEntity> products = sanPhamService.laySanPhamTheoListMaSP(imageNames);
	            model.addAttribute("products", products);
	            model.addAttribute("messageTrue", "Có thể bạn muốn tìm ");
	        }
	        xoaTepTinHinhAnh(avatarFileName);	        
	        
		return "deXuat/imageSearch";
	}
	
	public void xoaTepTinHinhAnh(String tenTep) {
	    String imgPath = filePath + tenTep; // Đường dẫn tới thư mục chứa hình ảnh
	    File tepTin = new File(imgPath);
	    
	    if (tepTin.exists()) {
	        tepTin.delete(); // Xóa tệp tin hình ảnh từ server
	    }
	}
}
