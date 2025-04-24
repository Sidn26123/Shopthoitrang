package ptithcm.controller;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.Period;
import java.time.ZoneId;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import javax.mail.internet.MimeMessage;
import javax.persistence.Id;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.hibernate.Hibernate;
import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.bean.Mailer;
import ptithcm.dao.nguoiDungDao;
import ptithcm.service.nguoiDungService;
import ptithcm.common.Constants;

@Transactional
@Controller
public class userController {

	@Autowired
	SessionFactory factory;

	@Autowired
	Mailer mailer;

	@Autowired
	nguoiDungService userService;

//	String uploadDir = "D:\\EclipseWorkSpace\\ShopThoiTrang\\src\\main\\webapp\\assets\\uploads";
	String uploadDir = Constants.FOLDER_UPLOAD_TEMP_IMG_FOR_SEARCH_PATH;
	@RequestMapping("user/login")
	public String login(ModelMap model) {
		model.addAttribute("user", new NguoiDungEntity());
		System.out.println("hello");
		String absolutePath = userController.class.getResource("").getPath();
		System.out.println("abs: " + absolutePath);
		return "/user/login";

	}

	@RequestMapping(value = "login", method = RequestMethod.GET)
	public String login0(ModelMap model) {
		model.addAttribute("user", new NguoiDungEntity());
		System.out.print("login");
		return "user/login";
	}

	@RequestMapping(value = "signup", method = RequestMethod.GET)
	public String signUp0(ModelMap model) {
		model.addAttribute("user", new NguoiDungEntity());
		System.out.print("signup");
		return "/user/signup";
	}

	@RequestMapping("forgotpass")
	public String forgotpass() {
		return "/user/forgotpass";
	}

	@RequestMapping("changePass.htm")
	public String changePass() {
		return "/user/changePass";
	}

	@RequestMapping("userInfo.htm")
	public String userInfo(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");

		model.addAttribute("user", user);

		return "/user/user-info";

	}

	@RequestMapping(value = "form/login", params = "login", method = RequestMethod.POST)
	public String login(ModelMap model, HttpServletRequest request, @ModelAttribute("user") NguoiDungEntity user,
			BindingResult errors) {

		Boolean loi = Boolean.TRUE;

		if (user.getUserName().isEmpty()) {
			errors.rejectValue("userName", "user", "Hãy nhập username hoặc email !!!");
			return "/user/login";
		} else if (user.getPassWord().isEmpty()) {
			errors.rejectValue("passWord", "user", "Hãy nhập mật khẩu !!!");
			return "/user/login";
		}

		NguoiDungEntity check = userService.findUserByNameAndEmail(user.getUserName(), user.getUserName());

		if (check == null) {

			errors.rejectValue("userName", "user", "Tài khoản không tồn tại !!!");
			loi = Boolean.FALSE;
			
		} else if ((user.getUserName().equals(check.getUserName()) || user.getUserName().equals(check.getEmail()))
				&& !userService.kiemTraMatKhau(user.getPassWord(), check.getPassWord())) {
			errors.rejectValue("passWord", "user", "Sai mật khẩu !!!");
			loi = Boolean.FALSE;
		}		
		else if (check.isTrangThai()==false) {
			errors.rejectValue("userName", "user", "Tài khoản của bạn đã bị khoá !!!");
			loi = Boolean.FALSE;
		}
		if (loi == Boolean.FALSE)
			return "user/login";
		Hibernate.initialize(check.getGioHangs());
		HttpSession session0 = request.getSession();
		session0.setAttribute("USER", check);		
		if (check.getQuyen()!=0) { // Kiểm tra quyền người dùng là admin
			return "redirect:/admin/index.htm"; // Trả về trang quản trị admin
		}		
		String maSp= (String) session0.getAttribute("SANPHAM");
		if(maSp!=null)
			{session0.removeAttribute("SANPHAM");
			return "redirect:/sanpham/"+maSp+".htm";}
		session0.removeAttribute("SANPHAM");
		return "redirect:/";
	}

	@RequestMapping(value = "form/signup", params = "signup", method = RequestMethod.POST)
	public String signUp(HttpServletRequest request, ModelMap model, @ModelAttribute("user") NguoiDungEntity user,
			BindingResult errors) throws ParseException {

		Boolean loi = Boolean.TRUE;
		String rePassword = request.getParameter("re-passWord");
		NguoiDungEntity userCheck;

		if (user.getEmail().isEmpty()) {
			errors.rejectValue("email", "user", "Hãy nhập email !!!");
			loi = Boolean.FALSE;
		} else if (!user.getEmail().endsWith("@gmail.com")) {
			errors.rejectValue("email", "user", "Hãy nhập email đúng định dạng !!!");
			loi = Boolean.FALSE;
		}
		if (user.getUserName().isEmpty()) {
			errors.rejectValue("userName", "user", "Hãy nhập username !!!");
			loi = Boolean.FALSE;
		} else if (user.getUserName().contains(" ")) {
			errors.rejectValue("userName", "user", "UserName không được chứa khoảng trắng !!!");
			loi = Boolean.FALSE;
		} else if (user.getUserName().length() > 30) {
			errors.rejectValue("userName", "user", "UserName không được dài quá 30 kí tự !!!");
			loi = Boolean.FALSE;
		}
		

		userCheck = userService.findUserByNameAndEmail(user.getUserName(), user.getEmail());
		if (userCheck != null) {
			if (userCheck.getEmail().equals(user.getEmail())) {
				errors.rejectValue("email", "user", "Email đã được sử dụng !!!");
				loi = Boolean.FALSE;
			}
			if (userCheck.getUserName().equals(user.getUserName())) {
				errors.rejectValue("userName", "user", "Username đã được sử dụng !!!");
				loi = Boolean.FALSE;
			}

		}

		if (user.getPassWord().isEmpty()) {
			errors.rejectValue("passWord", "user", "Hãy nhập mật khẩu !!!");
			loi = Boolean.FALSE;
		} else if (user.getPassWord().length() < 8) {
			errors.rejectValue("passWord", "user", "Password tối thiểu 8 kí tự !!!");
			loi = Boolean.FALSE;
		} else if (user.getPassWord().contains(" ")) {
			errors.rejectValue("passWord", "user", "Password không được chứa khoảng trắng !!!");
			loi = Boolean.FALSE;
		} else if (rePassword.isEmpty()) {
			errors.rejectValue("passWord", "user", "Hãy nhập lại mật khẩu !!!");
			loi = Boolean.FALSE;
		} else if (!rePassword.equals(user.getPassWord())) {
			errors.rejectValue("passWord", "user", "Xác nhận mật khẩu không đúng !!!");
			loi = Boolean.FALSE;
		}
		if (user.getHoTen().isEmpty()) {
			errors.rejectValue("hoTen", "user", "Hãy nhập họ tên !!!");
			loi = Boolean.FALSE;
		} else if (user.getHoTen().length() > 50) {
			errors.rejectValue("hoTen", "user", "Họ tên quá dài !!!");
			loi = Boolean.FALSE;
		} else if (!user.getHoTen().matches("[\\p{L} ]+")) {
			errors.rejectValue("hoTen", "user", "Họ tên không được chứa số !!!");
			loi = Boolean.FALSE;
		}

		if (user.getNgaySinh() == null) {
			errors.rejectValue("ngaySinh", "user", "Hãy nhập ngày sinh !!!");
			loi = Boolean.FALSE;
		} else if (!isValidAndOver18(user.getNgaySinh())) {
			errors.rejectValue("ngaySinh", "user", "Bạn cần lớn hơn 18 tuổi để tạo tài khoản !!!");
			loi = Boolean.FALSE;
		}
		if (user.getSdt().isEmpty()) {
			errors.rejectValue("sdt", "user", "Hãy nhập sdt !!!");
			loi = Boolean.FALSE;
		} else if (!user.getSdt().matches("[0-9]+")) {
			errors.rejectValue("sdt", "user", "SDT không hợp lệ !!!");
			loi = Boolean.FALSE;
		} else if (user.getSdt().length() != 10 && user.getSdt().length() != 11) {
			errors.rejectValue("sdt", "user", "SDT không hợp lệ !!!");
			loi = Boolean.FALSE;
		}
		if (user.getDiaChi().isEmpty()) {
			errors.rejectValue("diaChi", "user", "Hãy nhập địa chỉ !!!");
			loi = Boolean.FALSE;
		}

		if (loi == Boolean.FALSE)
			return "/user/signup";

		String NGAYSINH = request.getParameter("ngaySinh");
		java.sql.Date ns = java.sql.Date.valueOf(NGAYSINH);
		user.setPassWord(userService.maHoaMatKhau(user.getPassWord()));
		user.setNgaySinh(ns);
		user.setHoTen(capitalizeString(user.getHoTen()));
		user.setTrangThai(true);
		user.setQuyen(0);
		
		HttpSession session = request.getSession();
		session.setAttribute("USERSIGNUP", user);
		String otp = taoOTP();
		session.setAttribute("OTP", otp);
		mailer.sendMailAsync("SHOPTHOITRANG", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp); 
		return "/user/verify";
	}

	@RequestMapping(value = "form/forgotpass", params = "tieptheo", method = RequestMethod.POST)
	public String forgotPass(HttpServletRequest request, ModelMap model) {
		String userName = request.getParameter("username");
		System.out.print(userName);

		if (userName.equals(null)) {
			model.addAttribute("messenger", "Hãy nhập Username/Email của bạn !!!");
			return "/user/forgotpass";
		}
		NguoiDungEntity user = userService.findUserByNameAndEmail(userName, userName);
		if (user == null) {
			model.addAttribute("messenger", "Không tìm thấy tài khoản !!!");
			return "/user/forgotpass";
		}
		HttpSession session = request.getSession();
		session.setAttribute("USERFORGOT", user);
		String otp = taoOTP();
		session.setAttribute("OTP", otp);
		mailer.sendMailAsync("DAILYSHOP", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp);
		model.addAttribute("email","****"+user.getEmail().substring(user.getEmail().length()-13));
		return "/user/verify2";
	}

	@RequestMapping(value = "form/verify", params = "verify", method = RequestMethod.POST)
	public String verify(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		String a = request.getParameter("a");
		String b = request.getParameter("b");
		String c = request.getParameter("c");
		String d = request.getParameter("d");
		String e = request.getParameter("e");
		String f = request.getParameter("f");

		String otp = session.getAttribute("OTP").toString();
		String temp = a + b + c + d + e + f;

		if (otp.equals(temp)) {
			NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERSIGNUP"); 
			userService.addUser(user);
			model.addAttribute("user", new NguoiDungEntity());  //nếu nhập đúng OTP thì sẽ chuyển đến trang login
			return "/user/login";
		}
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERSIGNUP");
		model.addAttribute("email","****"+user.getEmail().substring(user.getEmail().length()-13));

		model.addAttribute("messenger", "OTP bạn nhập không đúng !!!");
		return "/user/verify";
	}

	@RequestMapping(value = "form/verify2", params = "verify", method = RequestMethod.POST)
	public String verify2(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		String a = request.getParameter("a");
		String b = request.getParameter("b");
		String c = request.getParameter("c");
		String d = request.getParameter("d");
		String e = request.getParameter("e");
		String f = request.getParameter("f");

		String otp = session.getAttribute("OTP").toString();
		String temp = a + b + c + d + e + f;

		if (otp.equals(temp)) {
			
			return "/user/newpass";
		}
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERFORGOT");
		model.addAttribute("email","****"+user.getEmail().substring(user.getEmail().length()-13));
		model.addAttribute("messenger", "OTP bạn nhập không đúng !!!");
		return "/user/verify2";
	}

	@RequestMapping(value = "form/verify", params = "again", method = RequestMethod.POST)
	public String guiLaiMa(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		String otp = taoOTP();
		session.setAttribute("OTP", otp);
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERSIGNUP");
//		sendMail("THEGIOIDIENMAY", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp);
		mailer.sendMailAsync("THEGIOIDIENMAY", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp);
		model.addAttribute("email","****"+user.getEmail().substring(user.getEmail().length()-13));
		model.addAttribute("again", "OTP đã được gửi lại !!!");
		return "/user/verify";
	}

	@RequestMapping(value = "form/verify2", params = "again", method = RequestMethod.POST)
	public String guiLaiMa2(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		String otp = taoOTP();
		session.setAttribute("OTP", otp);
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERFORGOT");
//		sendMail("THEGIOIDIENMAY", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp);
		mailer.sendMailAsync("THEGIOIDIENMAY", user.getEmail(), "OTP", "Mã OTP của bạn là: " + otp);
		model.addAttribute("email","****"+user.getEmail().substring(user.getEmail().length()-13));
		model.addAttribute("again", "OTP đã được gửi lại !!!");
		return "/user/verify2";
	}

	@RequestMapping(value = "form/newpass", params = "hoanTat", method = RequestMethod.POST)
	public String newPass(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();

		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USERFORGOT");
		String pass = request.getParameter("password");
		String confirmPass = request.getParameter("confirmPass");

		if (pass.isEmpty()) {
			model.addAttribute("messenger", "Mật khẩu không được trống !!!");
			return "/user/newpass";
		}
		if (!pass.equals(confirmPass)) {
			model.addAttribute("messenger", "Xác nhận mật khẩu không khớp !!!");
			return "/user/newpass";
		}
		if (pass.length() < 8) {
			model.addAttribute("messenger", "Mật khẩu phải hơn 8 kí tự !!!");
			return "/user/newpass";
		}
		if (pass.equals(confirmPass)) {

			user.setPassWord(userService.maHoaMatKhau(pass));

			userService.updateUser(user);

			model.addAttribute("user", new NguoiDungEntity());
			return "/user/login";
		}

		return "/user/newpass";
	}

	@RequestMapping("user/info")
	public String info(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");

		model.addAttribute("user", user);

		return "/user/user-info";
	}

	@RequestMapping(value = "form/info", params = "save", method = RequestMethod.POST)
	public String infoUpdate(HttpServletRequest request, ModelMap model, @ModelAttribute("user") NguoiDungEntity user,
			@RequestParam("image") MultipartFile file, BindingResult errors) {
		Boolean loi = Boolean.TRUE;
		HttpSession session = request.getSession();
		
		String currentWorkingDir = System.getProperty("user.dir");
		NguoiDungEntity userSave = (NguoiDungEntity) session.getAttribute("USER");

		String fileName = ""; //Thêm unix vào đầu file để tránh trùng tên file
		if (file.isEmpty() && userSave.getAvatar().isEmpty()) {
            // Handle empty file]
			fileName = "";
    		model.addAttribute("loiAvatar","File avatar không được để trống");
    		loi = Boolean.FALSE;
        }
		else if (file.isEmpty()) {
			fileName = "";
		}
        else {
        	fileName = System.currentTimeMillis() + "_" + file.getOriginalFilename();
            try {
                // Save file to the upload directory
            	System.out.println(uploadDir);
                File dest = new File(uploadDir + "\\" + fileName);
                file.transferTo(dest);

            
            } catch (IOException e) {
                e.printStackTrace();
        		model.addAttribute("loiAvatar","Có lỗi xảy ra!");

                loi = Boolean.FALSE;
            }
        }

		user.setUserName(userSave.getUserName());
		user.setEmail(userSave.getEmail());
		user.setSdt(userSave.getSdt());
		if (!(fileName.isBlank() || fileName.isEmpty())) {
			user.setAvatar(fileName);
		}


		String NGAYSINH = request.getParameter("ngaySinh");
		java.sql.Date ns = java.sql.Date.valueOf(NGAYSINH);
		if (user.getHoTen().isEmpty()) {
			model.addAttribute("loiHoTen", "Họ tên không được để trống !!!");

			loi = Boolean.FALSE;

		} else if (user.getHoTen().length() > 50)

		{
			model.addAttribute("loiHoTen", "Họ tên quá dài !!!");
			loi = Boolean.FALSE;
		} else if (!user.getHoTen().matches("[\\p{L} ]+")) {
			model.addAttribute("loiHoTen", "Họ tên không được chứa số !!!");
			loi = Boolean.FALSE;
		}

		if (user.getNgaySinh() == null) {
			model.addAttribute("loiNgaySinh", "Hãy nhập ngày sinh !!!");
			loi = Boolean.FALSE;
		} else if (!isValidAndOver18(user.getNgaySinh())) {
			model.addAttribute("loiNgaySinh", "Bạn cần lớn hơn 18 tuổi để tạo tài khoản !!!");
			loi = Boolean.FALSE;
		}
		if (user.getDiaChi().isEmpty()) {
			model.addAttribute("loiDiaChi", "Địa chỉ không được trống!!!");
			loi = Boolean.FALSE;
		}

		if (loi == Boolean.FALSE) {
			model.addAttribute("user", userSave);
			return "/user/user-info";
		}
		userSave.setHoTen(user.getHoTen());
		userSave.setGioiTinh(user.isGioiTinh());
		userSave.setNgaySinh(ns);
		userSave.setDiaChi(user.getDiaChi());
		if (!(fileName.isBlank() || fileName.isEmpty())) {
			userSave.setAvatar(fileName);
		}


		session.setAttribute("USER", userSave);
		
		userService.updateUser(userSave);

		model.addAttribute("user", userSave);
		model.addAttribute("successMessage","Cập nhật thông tin thành công");

		return "/user/user-info";
	}

	@RequestMapping(value = "form/changePass.htm", params = "save", method = RequestMethod.POST)
	public String changePass(HttpServletRequest request, ModelMap model) {
		Boolean loi = Boolean.TRUE;
		HttpSession session = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");
		String pass = request.getParameter("password");
		String newPass = request.getParameter("newPassword");
		String reNewPass = request.getParameter("reNewPassword");

		if (pass.isEmpty()) {
			model.addAttribute("loiPassword", "Hãy nhập mật khẩu cũ !!!");
			loi = Boolean.FALSE;

		} else if (!userService.kiemTraMatKhau(pass,user.getPassWord())) {
			model.addAttribute("loiPassword", "Mật khẩu cũ không đúng !!!");
			return "/user/user-info";
		}

		if (newPass.isEmpty()) {
			model.addAttribute("loiNewPassword", "Hãy nhập mật khẩu mới !!!");
			loi = Boolean.FALSE;
		} else if (newPass.length() < 8) {
			model.addAttribute("loiNewPassword", "Mật khẩu tối thiểu 8 kí tự !!!");
			loi = Boolean.FALSE;
		} else if (newPass.contains(" ")) {
			model.addAttribute("loiNewPassword", "Mật khẩu không được chứa khoảng trắng !!!");
			loi = Boolean.FALSE;
		} else if (reNewPass.isEmpty()) {
			model.addAttribute("loiRePassword", "Hãy nhập lại mật khẩu !!!");
			loi = Boolean.FALSE;
		} else if (!reNewPass.equals(newPass)) {
			model.addAttribute("loiRePassword", "Xác nhận mật khẩu không đúng !!!");
			loi = Boolean.FALSE;
		}

		if (loi == Boolean.TRUE) {
			user.setPassWord(userService.maHoaMatKhau(newPass));

			userService.updateUser(user);
			model.addAttribute("thanhCong", "Đổi mật khẩu thành công !!!");

		}
		
	

		model.addAttribute("user", user);

		return "/user/user-info";
	}

	@RequestMapping(value = "user/logout")
	public String logout(HttpServletRequest request, ModelMap model) {
		HttpSession session0 = request.getSession();
		session0.removeAttribute("USER");
		session0.removeAttribute("SANPHAM");
		session0.removeAttribute("NEWINFO");
		 return "redirect:/";

	}

	public List<SanPhamEntity> laySanPhamTheoLoai(String loai) {
		Session session = factory.getCurrentSession();
		String hql = "FROM SanPhamEntity sp WHERE sp.loaiSanPham.maLoai = :loai ";
		Query query = session.createQuery(hql).setParameter("loai", loai);
		query.setMaxResults(6);
		List<SanPhamEntity> list = query.list();
		return list;
	}

	public boolean isValidAndOver18(Date ngaySinh) {
		LocalDate dob = ngaySinh.toInstant().atZone(ZoneId.systemDefault()).toLocalDate();
		LocalDate today = LocalDate.now();
		int age = Period.between(dob, today).getYears();
		Date currentDate = new Date();
		return !ngaySinh.after(currentDate) && age >= 18;
	}

	public static String capitalizeString(String str) {
		String[] words = str.split("\\s+");
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].substring(0, 1).toUpperCase() + words[i].substring(1);
		}
		return String.join(" ", words);
	}

	public String taoOTP() {
		String alphabelt = "0123456789qwertyuiopasdfghjkzxcvbnmQWERTYUOPLKJHGFDSAZXCVBNM";

		String otp = "";
		Random random = new Random();
		for (int i = 0; i < 6; i++) {
			otp += alphabelt.charAt(random.nextInt(60));
		}

		return otp;
	}


}