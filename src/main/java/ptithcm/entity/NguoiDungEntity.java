package ptithcm.entity;

import java.util.Date;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.persistence.Temporal;
import javax.persistence.TemporalType;

import org.springframework.format.annotation.DateTimeFormat;

@Entity
@Table(name = "NGUOIDUNG")
public class NguoiDungEntity {
	@Id
	@Column(name = "MAND")
	@GeneratedValue
	private Integer maNd;
	@Column(name = "USERNAME")
	private String userName;
	@Column(name = "PASSWORD")
	private String passWord;
	@Column(name = "HOTEN")
	private String hoTen;
	@Column(name = "GIOITINH")
	private boolean gioiTinh;
	@Temporal(TemporalType.DATE)
	@DateTimeFormat(pattern = "yyyy-mm-dd")
	@Column(name = "NGAYSINH")
	private Date ngaySinh;
	@Column(name = "SDT")
	private String sdt;
	@Column(name = "EMAIL")
	private String email;
	@Column(name = "AVATAR")
	private String avatar;
	@Column(name = "DIACHI")
	private String diaChi;
	@Column(name = "QUYEN")
	private int quyen;
	@Column(name = "TRANGTHAI")
	private boolean trangThai;
	@OneToMany(mappedBy = "nguoiDung", fetch = FetchType.LAZY)
	private List<GioHangEntity> gioHangs;
	@OneToMany(mappedBy = "nguoiDung", fetch = FetchType.LAZY)
	private List<DanhGiaEntity> danhGiaList;
	@OneToMany(mappedBy = "nguoiDung", fetch = FetchType.LAZY)
	private List<DonHangEntity> donHangs;
	@OneToMany(mappedBy = "nguoiDung", fetch = FetchType.LAZY)
	private List<YeuThichEntity> yeuThichList;
	@OneToMany(mappedBy = "nguoiDung", fetch = FetchType.LAZY)
	private List<ThongBaoEntity> thongBaoList;
	
	public List<YeuThichEntity> getYeuThichList() {
		return yeuThichList;
	}

	public void setYeuThichList(List<YeuThichEntity> yeuThichList) {
		this.yeuThichList = yeuThichList;
	}

	public List<DonHangEntity> getDonHangs() {
		return donHangs;
	}

	public void setDonHangs(List<DonHangEntity> donHangs) {
		this.donHangs = donHangs;
	}

	

	public Integer getMaNd() {
		return maNd;
	}

	public void setMaNd(Integer maNd) {
		this.maNd = maNd;
	}

	public List<GioHangEntity> getGioHangs() {
		return gioHangs;
	}

	public void setGioHangs(List<GioHangEntity> gioHangs) {
		this.gioHangs = gioHangs;
	}

	public List<DanhGiaEntity> getDanhGiaList() {
		return danhGiaList;
	}

	public void setDanhGiaList(List<DanhGiaEntity> danhGiaList) {
		this.danhGiaList = danhGiaList;
	}

	public List<ThongBaoEntity> getThongbao(){
		return thongBaoList;
	}
	
	public void setThongBaoList(List<ThongBaoEntity> thongBaoList) {
		this.thongBaoList = thongBaoList;
	}
	
	public String getUserName() {
		return userName;
	}

	public void setUserName(String userName) {
		this.userName = userName;
	}

	public String getPassWord() {
		return passWord;
	}

	public void setPassWord(String passWord) {
		this.passWord = passWord;
	}

	public String getHoTen() {
		return hoTen;
	}

	public void setHoTen(String hoTen) {
		this.hoTen = hoTen;
	}

	public boolean isGioiTinh() {
		return gioiTinh;
	}

	public void setGioiTinh(boolean gioiTinh) {
		this.gioiTinh = gioiTinh;
	}

	public Date getNgaySinh() {
		return ngaySinh;
	}

	public void setNgaySinh(Date ngaySinh) {
		this.ngaySinh = ngaySinh;
	}

	public String getSdt() {
		return sdt;
	}

	public void setSdt(String sdt) {
		this.sdt = sdt;
	}

	public String getEmail() {
		return email;
	}

	public void setEmail(String email) {
		this.email = email;
	}

	public String getDiaChi() {
		return diaChi;
	}

	public void setDiaChi(String diaChi) {
		this.diaChi = diaChi;
	}

	public int getQuyen() {
		return quyen;
	}

	public void setQuyen(int quyen) {
		this.quyen = quyen;
	}

	public boolean isTrangThai() {
		return trangThai;
	}

	public void setTrangThai(boolean trangThai) {
		this.trangThai = trangThai;
	}
	
	public String getAvatar() {
		return avatar;
	}

	public void setAvatar(String avatar) {
		this.avatar = avatar;
	}
	
	

}