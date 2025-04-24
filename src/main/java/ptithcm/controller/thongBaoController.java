package ptithcm.controller;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

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

import ptithcm.entity.DanhGiaEntity;
import ptithcm.entity.GioHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.YeuThichEntity;
import ptithcm.service.DanhGiaService;
import ptithcm.service.SanPhamService;
import ptithcm.service.gioHangService;
import ptithcm.service.yeuThichService;

import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.ThongBaoEntity;
import ptithcm.entity.LoaiSanPhamEntity;
import ptithcm.service.DonHangService;
import ptithcm.service.SanPhamService;
import ptithcm.service.loaiSanPhamService;
import ptithcm.service.thongBaoService;


@Controller
public class thongBaoController {
	
	@Autowired
	thongBaoService tbService;
	
	
    @RequestMapping("notifications/readNoti")
    public String resetNotifications(HttpServletRequest request, ModelMap model) {
        // Lấy người dùng từ session
        NguoiDungEntity nd = (NguoiDungEntity) request.getSession().getAttribute("USER");

        // Nếu người dùng tồn tại, đặt số lượng thông báo chưa đọc về 0
        if (nd != null && nd.getMaNd() > 0) {
            List<ThongBaoEntity> listThongBao = tbService.LayThongBaoCuaUser(nd.getMaNd());
            
            // Đặt tất cả thông báo thành "đã đọc"
            tbService.markAllNotificationRead(nd.getMaNd());
            
            // Cập nhật giá trị notificationsCount
            model.addAttribute("notificationsCount", 0);
        }

        return "success"; // Hoặc trả về thông tin gì đó nếu cần
    }
}
